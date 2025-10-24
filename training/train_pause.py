import random
import time
import numpy as np
import torch
from tqdm import tqdm
from models.pause import Pause
from utils.utils import clear_cache_in_dict, evaluate_pred


def train_pause_model(logger, args, train_dataset, eval_dataset, lr=5e-5, wd=0.01):
    num_epochs = args.cg_linear_epochs + args.cg_llm_epochs
    pause = Pause(
        args.teacher_model_name,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.config,
    ).to(args.device)
    optimizer = torch.optim.AdamW(pause.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        pause.train()
        train_loss, val_loss, batch_loss = 0, 0, 0
        for i, idx in enumerate(tqdm(
            random.sample(range(len(train_dataset)), len(train_dataset)),
            desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        )):
            input_ids, attn_mask, label = pause.process_sample(
                train_dataset[idx],
                args.max_seq_len,
                args.device,
                args.train_max_contemp_tokens,
            )
            input_embs = pause.model.get_input_embeddings()(input_ids)
            outputs = pause.model(
                inputs_embeds=input_embs, attention_mask=attn_mask, labels=label
            )
            batch_loss += outputs.loss
            clear_cache_in_dict(outputs)
            if ((i + 1) % args.batch_size == 0) or i == len(train_dataset) - 1:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
                batch_loss = 0
            torch.cuda.empty_cache()
        avg_train_loss = train_loss / len(train_dataset)

        pause.eval()
        with torch.no_grad():
            for sample in tqdm(
                eval_dataset, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                input_ids, attn_mask, label = pause.process_sample(
                    sample,
                    args.max_seq_len,
                    args.device,
                    args.train_max_contemp_tokens,
                )
                input_embs = pause.model.get_input_embeddings()(input_ids)
                outputs = pause.model(
                    inputs_embeds=input_embs, attention_mask=attn_mask, labels=label
                )
                val_loss += outputs.loss.item()
                clear_cache_in_dict(outputs)
            torch.cuda.empty_cache()
        avg_val_loss = val_loss / len(eval_dataset)
        optimizer.zero_grad()
        del outputs, batch_loss, input_ids, attn_mask, label

        # Log metrics
        logger.log_metrics(
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, epoch
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            pause.save_pretrained(args.model_save_path)
            logger.logger.info(
                f"Saved best model with validation loss: {best_val_loss:.6f}"
            )
    del pause
    torch.cuda.empty_cache()
    return args.model_save_path


def run_pause_inference(logger, pause, dataset, args):
    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                input_ids, attn_mask, _ = pause.process_sample(
                    sample,
                    args.max_seq_len,
                    args.device,
                    args.eval_max_contemp_tokens,
                    append_ans=False
                )
                input_embs = pause.model.get_input_embeddings()(input_ids)

                gen_start = time.time()
                outputs = pause.model.generate(
                    inputs_embeds=input_embs,
                    attention_mask=attn_mask,
                    max_length=30 + input_ids.shape[-1],
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=pause.tokenizer.eos_token_id,
                )
                gen_time = time.time() - gen_start
                prefix_length = (
                    input_ids.shape[1] - 1 if len(outputs[0]) > input_ids.shape[1] else 0
                )
                answer = pause.tokenizer.decode(
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
                del input_ids, attn_mask, outputs
                torch.cuda.empty_cache()
            metrics = {
                    "numerical_accuracy": float(np.mean([r["correct"] for r in results])),
                    "ave_sample_time": float(np.mean([r["sample_time"] for r in results])),
                    "dataset": args.dataset,
                    "eval_temp": temp,
                    "train_max_contemp_tokens": args.train_max_contemp_tokens,
                    "eval_max_contemp_tokens": args.eval_max_contemp_tokens,
                }
            logger.logger.info(f"eval_temp = {temp} | acc = {metrics['numerical_accuracy']} | ave_sample_time = {metrics['ave_sample_time']}")
            all_metrics.append(metrics)
    return all_metrics