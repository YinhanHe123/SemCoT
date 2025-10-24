import random
import time
import numpy as np
import torch
from tqdm import tqdm
from models.icot_si import ICoT_SI
from utils.utils import clear_cache_in_dict, evaluate_pred


def train_icot_si_model(logger, args, train_dataset, eval_dataset, lr=1e-5, remove_per_epoch=8):
    num_epochs = args.cg_linear_epochs + args.cg_llm_epochs
    icot_si = ICoT_SI(
        args.teacher_model_name,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.config,
    ).to(args.device)
    optimizer = torch.optim.AdamW(icot_si.parameters(), lr=lr)
    best_val_loss = float("inf")
    tok_to_remove, masked_complete_cot = 0, False
    for epoch in range(num_epochs):
        icot_si.train()
        train_loss, val_loss, batch_loss = 0, 0, 0
        for i, idx in enumerate(
            tqdm(
                random.sample(range(len(train_dataset)), len(train_dataset)),
                desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
            )
        ):
            if i % (len(train_dataset) // remove_per_epoch) == 0:
                tok_to_remove += 1
                if not masked_complete_cot:
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    optimizer = torch.optim.AdamW(icot_si.parameters(), lr=lr)
            input_ids, attn_mask, pos_ids, labels, masked_complete_cot = (
                icot_si.process_sample(
                    train_dataset[idx],
                    args.max_seq_len,
                    args.device,
                    tok_to_remove,
                )
            )
            if not masked_complete_cot:
                best_val_loss = float("inf")
            input_embs = icot_si.model.get_input_embeddings()(input_ids)
            outputs = icot_si.model(
                inputs_embeds=input_embs,
                attention_mask=attn_mask,
                labels=labels,
                position_ids=pos_ids,
            )
            batch_loss += outputs.loss
            clear_cache_in_dict(outputs)
            if ((i + 1) % args.batch_size == 0) or i == len(train_dataset) - 1:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
                del input_ids, input_embs, attn_mask, pos_ids, labels, outputs, batch_loss
                batch_loss = 0
            torch.cuda.empty_cache()
        avg_train_loss = train_loss / len(train_dataset)

        icot_si.eval()
        with torch.no_grad():
            for sample in tqdm(
                eval_dataset, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                input_ids, attn_mask, pos_ids, labels, _ = icot_si.process_sample(
                    sample,
                    args.max_seq_len,
                    args.device,
                    tok_to_remove,
                )
                input_embs = icot_si.model.get_input_embeddings()(input_ids)
                outputs = icot_si.model(
                    inputs_embeds=input_embs,
                    attention_mask=attn_mask,
                    labels=labels,
                    position_ids=pos_ids,
                )
                val_loss += outputs.loss.item()
                clear_cache_in_dict(outputs)
            torch.cuda.empty_cache()
        avg_val_loss = val_loss / len(eval_dataset)
        optimizer.zero_grad()
        del input_ids, attn_mask, pos_ids, labels, outputs, input_embs

        # Log metrics
        logger.log_metrics(
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, epoch
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            icot_si.save_pretrained(args.model_save_path)
            logger.logger.info(
                f"Saved best model with validation loss: {best_val_loss:.6f}"
            )
    del icot_si
    torch.cuda.empty_cache()
    return args.model_save_path


def run_icot_si_inference(logger, icot_si, dataset, args):
    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                input_ids, attn_mask, pos_ids, labels, _ = icot_si.process_sample(
                    sample, args.max_seq_len, args.device, append_ans=False
                )
                input_embs = icot_si.model.get_input_embeddings()(input_ids)

                gen_start = time.time()
                outputs = icot_si.model.generate(
                    inputs_embeds=input_embs,
                    attention_mask=attn_mask,
                    max_length=30 + input_ids.shape[-1],
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=None,
                    eos_token_id=-1,
                    logits_processor=icot_si.logits_processor,
                    stopping_criteria=icot_si.stopping_criteria,
                )
                gen_time = time.time() - gen_start
                prefix_length = (
                    input_ids.shape[1] - 1 if len(outputs[0]) > input_ids.shape[1] else 0
                )
                answer = icot_si.tokenizer.decode(
                    outputs[0][prefix_length:], skip_special_tokens=True
                )
                results.append(
                    {
                        "query": sample["query"],
                        "correct": int(
                            evaluate_pred(answer, sample["answer"], dataset.name)
                        ),
                        "sample_time": gen_time,
                    }
                )
                del input_ids, attn_mask, outputs, pos_ids, labels, input_embs
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
