import random
import time
import numpy as np
import torch
from tqdm import tqdm
from models.coconut import Coconut
from utils.utils import evaluate_pred, clear_cache_in_dict


def train_coconut_model(
    logger,
    args,
    train_dataset,
    eval_dataset,
    lr=1e-5,
    wd=0.01,
    num_stages=3,
):
    num_epochs = (args.cg_linear_epochs + args.cg_llm_epochs) // num_stages
    coconut = Coconut(
        args.teacher_model_name,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.config,
    ).to(args.device)
    best_loss = float("inf")
    for stage in range(num_stages):
        for epoch in range(num_epochs):
            logger.logger.info(f"Stage {stage + 1}, Epoch {epoch + 1}/{num_epochs}")
            optimizer = torch.optim.AdamW(coconut.parameters(), lr=lr, weight_decay=wd)
            coconut.train()
            train_loss, batch_loss, eval_loss = 0, 0, 0
            for i, idx in enumerate(
                tqdm(
                    random.sample(range(len(train_dataset)), len(train_dataset)),
                    desc=f"Stage {stage + 1}, Epoch {epoch + 1}/{num_epochs} - Training",
                )
            ):
                input_ids, attn_mask, label, bot_idx, eot_idx = coconut.process_sample(
                    train_dataset[idx],
                    args.max_seq_len,
                    args.device,
                    args.train_max_contemp_tokens,
                    stage,
                )
                input_embs = coconut.model.get_input_embeddings()(input_ids)
                new_inputs, outputs = coconut(
                    input_embs, attn_mask, bot_idx, eot_idx, label
                )
                batch_loss += outputs.loss
                clear_cache_in_dict(outputs)
                if ((i + 1) % args.batch_size == 0) or i == len(train_dataset) - 1:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    train_loss += batch_loss.item()
                    del (
                        input_ids,
                        attn_mask,
                        label,
                        input_embs,
                        outputs,
                        batch_loss,
                        new_inputs,
                    )
                    torch.cuda.empty_cache()
                    batch_loss = 0
            avg_train_loss = train_loss / len(train_dataset)

            coconut.eval()
            with torch.no_grad():
                for sample in tqdm(
                    eval_dataset,
                    desc=f"Stage {stage + 1}, Epoch {epoch + 1}/{num_epochs} - Evaluation",
                ):
                    input_ids, attn_mask, label, bot_idx, eot_idx = (
                        coconut.process_sample(
                            sample,
                            args.max_seq_len,
                            args.device,
                            args.train_max_contemp_tokens,
                            stage,
                        )
                    )
                    input_embs = coconut.model.get_input_embeddings()(input_ids)
                    new_inputs, outputs = coconut(
                        input_embs, attn_mask, bot_idx, eot_idx, label
                    )
                    eval_loss += outputs.loss.item()
                    clear_cache_in_dict(outputs)
                    del input_ids, attn_mask, label, input_embs, new_inputs, outputs
                    torch.cuda.empty_cache()
            avg_eval_loss = eval_loss / len(eval_dataset)
            logger.log_metrics(
                {"train_loss": avg_train_loss, "val_loss": avg_eval_loss}, epoch
            )
            optimizer.zero_grad()
            del optimizer
            torch.cuda.empty_cache()

            # Save best model
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                coconut.save_pretrained(args.model_save_path)
                logger.logger.info(f"Saved best model with eval loss: {best_loss:.6f}")
    coconut = coconut.cpu()
    del coconut
    torch.cuda.empty_cache()
    return args.model_save_path


def run_coconut_inference(logger, coconut, dataset, args):
    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                input_ids, attn_mask, label, bot_idx, eot_idx = coconut.process_sample(
                    sample, args.max_seq_len, args.device, args.eval_max_contemp_tokens
                )
                input_embs = coconut.model.get_input_embeddings()(input_ids)

                contemp_start = time.time()
                new_inputs, outputs = coconut(
                    input_embs, attn_mask, bot_idx, eot_idx, label
                )
                contemp_time = time.time() - contemp_start

                # Generate text with continuous thoughts
                gen_start = time.time()
                outputs = coconut.model.generate(
                    inputs_embeds=new_inputs,
                    attention_mask=attn_mask,
                    max_length=30 + new_inputs.size(1),
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=coconut.tokenizer.eos_token_id,
                )
                gen_time = time.time() - gen_start

                ans_pos = (
                    new_inputs.shape[1] - 1
                    if outputs.shape[-1] > new_inputs.shape[1]
                    else 0
                )
                answer = coconut.tokenizer.decode(
                    outputs[0][ans_pos:], skip_special_tokens=True
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
                del input_ids, attn_mask, label, input_embs, new_inputs, outputs
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
