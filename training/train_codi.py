import random
import time
import numpy as np
import torch
from tqdm import tqdm
from models.codi import CODI
from utils.utils import clear_cache_in_dict, evaluate_pred
import torch.nn.functional as F


def train_codi_model(
    logger,
    args,
    train_dataset,
    eval_dataset,
    lr=8e-4,
    wd=0.01,
    alpha=1.0,
    beta=1.0,
    gamma=20.0,
):
    num_epochs = args.cg_linear_epochs + args.cg_llm_epochs
    codi = CODI(
        args.teacher_model_name,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.config,
    ).to(args.device)
    optimizer = torch.optim.AdamW(codi.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        codi.train()
        train_loss, val_loss, teacher_loss, student_loss, kd_loss = 0, 0, 0, 0, 0
        for idx in tqdm(
            random.sample(range(len(train_dataset)), len(train_dataset)),
            desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        ):
            optimizer.zero_grad()
            query, ans_prompt, cot, ans = codi.process_sample(
                train_dataset[idx], args.max_seq_len, args.device
            )
            teacher_outputs, teacher_idx = codi.forward_teacher(
                query, cot, ans, ans_prompt
            )
            student_outputs, cont_tokens, student_idx, begin_inputs, end_inputs = (
                codi.forward_student(
                    query, ans_prompt, args.train_max_contemp_tokens, ans
                )
            )
            sample_kd_loss = sum(
                [
                    F.smooth_l1_loss(
                        t.detach()[:, teacher_idx - 1 : -1],
                        s[:, student_idx - 1 : -1],
                    )
                    for t, s in zip(
                        teacher_outputs.hidden_states, student_outputs.hidden_states
                    )
                ]
            )
            sample_loss = (
                alpha * teacher_outputs.loss
                + beta * student_outputs.loss
                + gamma * sample_kd_loss
            )
            sample_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += sample_loss.item() / len(train_dataset)
            teacher_loss += teacher_outputs.loss.item() / len(train_dataset)
            student_loss += student_outputs.loss.item() / len(train_dataset)
            kd_loss += sample_kd_loss.item() / len(train_dataset)
            clear_cache_in_dict(teacher_outputs)
            clear_cache_in_dict(student_outputs)
            del sample_kd_loss, sample_loss, teacher_outputs, student_outputs
            del query, ans_prompt, cot, ans, cont_tokens, begin_inputs, end_inputs
            torch.cuda.empty_cache()
        logger.log_metrics(
            {
                "train/loss": train_loss,
                "train/teacher_loss": teacher_loss,
                "train/student_loss": student_loss,
                "train/kd_loss": kd_loss,
            },
            epoch,
        )

        codi.eval()
        with torch.no_grad():
            for sample in tqdm(
                eval_dataset, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                query, ans_prompt, cot, ans = codi.process_sample(
                    sample, args.max_seq_len, args.device
                )
                teacher_outputs, teacher_idx = codi.forward_teacher(
                    query, cot, ans, ans_prompt
                )
                student_outputs, cont_tokens, student_idx, begin_inputs, end_inputs = (
                    codi.forward_student(
                        query, ans_prompt, args.train_max_contemp_tokens, ans
                    )
                )
                sample_kd_loss = sum(
                    [
                        F.smooth_l1_loss(
                            t.detach()[:, teacher_idx - 1 : -1],
                            s[:, student_idx - 1 : -1],
                        )
                        for t, s in zip(
                            teacher_outputs.hidden_states, student_outputs.hidden_states
                        )
                    ]
                )
                sample_loss = (
                    alpha * teacher_outputs.loss
                    + beta * student_outputs.loss
                    + gamma * sample_kd_loss
                )
                val_loss += sample_loss.item() / len(eval_dataset)
                teacher_loss += teacher_outputs.loss.item() / len(eval_dataset)
                student_loss += student_outputs.loss.item() / len(eval_dataset)
                kd_loss += sample_kd_loss.item() / len(eval_dataset)
                clear_cache_in_dict(teacher_outputs)
                clear_cache_in_dict(student_outputs)
                del sample_kd_loss, sample_loss, teacher_outputs, student_outputs
                del query, ans_prompt, cot, ans, cont_tokens, begin_inputs, end_inputs
                torch.cuda.empty_cache()
        logger.log_metrics(
            {
                "eval/loss": val_loss,
                "eval/teacher_loss": teacher_loss,
                "eval/student_loss": student_loss,
                "eval/kd_loss": kd_loss,
            },
            epoch,
        )
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            codi.save_pretrained(args.model_save_path)
            logger.logger.info(
                f"Saved best model with validation loss: {best_val_loss:.6f}"
            )
    codi = codi.cpu()
    del codi
    torch.cuda.empty_cache()
    return args.model_save_path


def run_codi_inference(logger, codi, dataset, args):
    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                query, ans_prompt, _, ans = codi.process_sample(
                    sample, args.max_seq_len, args.device
                )
                contemp_start = time.time()
                student_out, cont_tokens, _, begin_input, end_input = (
                    codi.forward_student(
                        query, ans_prompt, args.eval_max_contemp_tokens, ans
                    )
                )
                contemp_time = time.time() - contemp_start

                begin_embs = codi.model.get_input_embeddings()(begin_input)
                end_embs = codi.model.get_input_embeddings()(end_input)
                input_embs = torch.cat([begin_embs, cont_tokens, end_embs], dim=1)
                attn_mask = torch.ones(input_embs.shape[0], input_embs.shape[1]).to(
                    args.device
                )

                gen_start = time.time()
                outputs = codi.model.generate(
                    inputs_embeds=input_embs,
                    attention_mask=attn_mask,
                    max_length=30 + input_embs.shape[1],
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=codi.tokenizer.eos_token_id,
                )
                gen_time = time.time() - gen_start
                prefix_length = (
                    input_embs.shape[1] - 1 if len(outputs[0]) > input_embs.shape[1] else 0
                )
                answer = codi.tokenizer.decode(
                    outputs[0][prefix_length:], skip_special_tokens=True
                )
                results.append(
                    {
                        "query": sample["query"],
                        "correct": int(
                            evaluate_pred(answer, sample["answer"], dataset.name)
                        ),
                        "sample_time": contemp_time + gen_time,
                    }
                )
                clear_cache_in_dict(student_out)
                del query, ans_prompt, ans, cont_tokens, begin_input, end_input
                del begin_embs, end_embs, input_embs, attn_mask, outputs, student_out
                torch.cuda.empty_cache()
            metrics = {
                "numerical_accuracy": float(np.mean([r["correct"] for r in results])),
                "ave_sample_time": float(np.mean([r["sample_time"] for r in results])),
                "dataset": args.dataset,
                "eval_temp": temp,
                "train_max_contemp_tokens": args.train_max_contemp_tokens,
                "eval_max_contemp_tokens": args.eval_max_contemp_tokens,
            }
            logger.logger.info(
                f"eval_temp = {temp} | acc = {metrics['numerical_accuracy']} | ave_sample_time = {metrics['ave_sample_time']}"
            )
            all_metrics.append(metrics)
    return all_metrics
