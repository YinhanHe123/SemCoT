import random
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from models.softcot import SoftCoT
from utils.utils import clear_cache_in_dict, evaluate_pred


def process_item(softcot, item, args):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # Generate soft thought tokens
    soft_thoughts = softcot.generate_soft_thoughts(
        item["query"], args.train_max_contemp_tokens, args.max_seq_len, args.device
    )
    # Project soft thoughts to LLM space
    llm_input, labels = softcot.get_combined_inputs(
        soft_thoughts, item, args.max_seq_len, args.device
    )
    # Generate answer
    attn_mask = torch.ones((1, llm_input.shape[1])).long().to(args.device)
    pos_ids = torch.arange(llm_input.shape[1]).long().to(args.device).unsqueeze(0)
    outputs = softcot.llm_model(
        inputs_embeds=llm_input,
        attention_mask=attn_mask,
        position_ids=pos_ids,
    )
    loss = criterion(outputs["logits"][:, :-1].squeeze(), labels[:, 1:].squeeze())
    clear_cache_in_dict(outputs)
    del soft_thoughts, llm_input, labels, attn_mask, pos_ids, outputs
    return loss


def train_softcot_model(logger, args, train_dataset, eval_dataset, lr=1e-4, wd=0.01):
    """
    Train the SoftCoT model's projection module.
    """
    num_epochs = args.cg_linear_epochs + args.cg_llm_epochs
    softcot = SoftCoT(args.config, args.teacher_model_name, args.student_model_name).to(
        args.device
    )
    optimizer = optim.AdamW(softcot.proj.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        softcot.proj.train()
        train_loss, val_loss, batch_loss = 0, 0, 0
        for i, idx in enumerate(
            tqdm(
                random.sample(range(len(train_dataset)), len(train_dataset)),
                desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
            )
        ):
            batch_loss += process_item(softcot, train_dataset[idx], args)
            if ((i + 1) % args.batch_size == 0) or i == len(train_dataset) - 1:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
                del batch_loss
                torch.cuda.empty_cache()
                batch_loss = 0
        avg_train_loss = train_loss / len(train_dataset)

        softcot.proj.eval()
        with torch.no_grad():
            for sample in tqdm(
                eval_dataset,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
            ):
                val_loss += process_item(softcot, sample, args)
            torch.cuda.empty_cache()
        avg_val_loss = val_loss / len(eval_dataset)

        # Log metrics
        logger.log_metrics(
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, epoch
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save checkpoint
            softcot.save_pretrained(args.model_save_path)
            logger.logger.info(
                f"Saved best model with validation loss: {best_val_loss:.6f}"
            )
    del softcot
    torch.cuda.empty_cache()
    return args.model_save_path


def run_softcot_inference(logger, softcot, dataset, args):
    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                contemp_start = time.time()
                soft_thoughts = softcot.generate_soft_thoughts(
                    sample["query"],
                    args.eval_max_contemp_tokens,
                    args.max_seq_len,
                    args.device,
                )
                contemp_time = time.time() - contemp_start
                # Project soft thoughts to LLM space
                llm_input, _ = softcot.get_combined_inputs(
                    soft_thoughts,
                    sample,
                    args.max_seq_len,
                    args.device,
                    append_ans=False,
                )
                # Generate answer
                attn_mask = torch.ones((1, llm_input.shape[1])).long().to(args.device)
                gen_start = time.time()
                outputs = softcot.llm_model.generate(
                    inputs_embeds=llm_input,
                    attention_mask=attn_mask,
                    max_length=30 + llm_input.size(1),
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=softcot.llm_tokenizer.eos_token_id,
                )
                gen_time = time.time() - gen_start
                prefix_length = (
                    llm_input.shape[1] - 1 if len(outputs[0]) > llm_input.shape[1] else 0
                )
                answer = softcot.llm_tokenizer.decode(
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
                del soft_thoughts, llm_input, attn_mask, outputs
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
