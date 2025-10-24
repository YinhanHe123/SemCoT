import gc
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch.optim as optim
from models.semcot import ContempGen, CustomST
from torch.utils.data import random_split
import torch.nn.functional as F
from utils.utils import clear_cache_in_dict, evaluate_pred, get_prompts


def load_reasoning_hidden(dataset, base_model_name, device, max_seq_len, layer_idx=16):
    base_model = AutoModel.from_pretrained(base_model_name).to(device)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    for idx, sample in enumerate(tqdm(dataset)):
        with torch.no_grad():
            orig_inputs = tokenizer(
                sample["reasoning"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            orig_outputs = base_model(**orig_inputs, output_hidden_states=True)
            dataset.update_item(
                idx, "gt_reason_hid", orig_outputs.hidden_states[layer_idx].cpu()
            )
            if "condensed_reasoning" in sample:
                cond_inputs = tokenizer(
                    sample["condensed_reasoning"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                ).to(device)
                cond_outputs = base_model(**cond_inputs, output_hidden_states=True)

                dataset.update_item(
                    idx, "cond_reason_hid", cond_outputs.hidden_states[layer_idx].cpu()
                )
                del cond_inputs, cond_outputs
    base_model = base_model.cpu()
    del base_model, orig_inputs, orig_outputs
    torch.cuda.empty_cache()


def update_reasoning_hidden(dataset, device, custom_st):
    for idx, sample in enumerate(tqdm(dataset)):
        gt_reason_hid = sample["gt_reason_hid"].to(device)
        gt_reason_hid = custom_st(gt_reason_hid).cpu()
        dataset.update_item(idx, "gt_reason_hid", gt_reason_hid)
    del gt_reason_hid
    torch.cuda.empty_cache()


def contrastive_loss(original_embeddings, condensed_embeddings):
    """
    Compute contrastive loss between original and condensed embeddings

    Args:
        original_embeddings: Embeddings for original reasoning
        condensed_embeddings: Embeddings for condensed reasoning

    Returns:
        Contrastive loss value
    """
    # Compute similarity matrix
    norm_original = torch.norm(original_embeddings, dim=1, keepdim=True)
    norm_condensed = torch.norm(condensed_embeddings, dim=1, keepdim=True)
    norm_mat = torch.matmul(norm_original, norm_condensed.T)
    sim_mat = torch.matmul(original_embeddings, condensed_embeddings.T) / norm_mat
    # Contrastive loss calculation
    loss = -(sim_mat.diag().exp() / sim_mat.exp().sum(dim=1)).log().mean()
    del norm_original, norm_condensed, norm_mat, sim_mat
    torch.cuda.empty_cache()
    return loss


def train_custom_st(dataset, args, logger):
    """
    Train a customized sentence transformer to measure similarity between
    reasoning pairs (original and condensed reasoning)
    """
    logger.logger.info("Training sentence transformer")
    device = args.device
    model_path = f"{args.model_save_path}/customst_llr={args.st_linear_lr}_lwd={args.st_linear_wd}_le={args.st_linear_epochs}_llmlr={args.st_llm_lr}_llmwd={args.st_llm_wd}_llme={args.st_llm_epochs}_traintok={args.train_max_contemp_tokens}"

    if args.variation == "param_analysis":
        model_path = model_path.replace("param_analysis", "vanilla")
        if os.path.exists(model_path):
            return model_path
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    sentence_transformer = CustomST(args.teacher_model_name).to(device)
    for name, param in sentence_transformer.named_parameters():
        if "embedding_projection" not in name:
            param.requires_grad = False
    del param
    for lr, wd, ne in [
        (args.st_linear_lr, args.st_linear_wd, args.st_linear_epochs),
        (args.st_llm_lr, args.st_llm_wd, args.st_llm_epochs),
    ]:
        best_val_loss = float("inf")
        optimizer = optim.AdamW(
            sentence_transformer.parameters(), lr=lr, weight_decay=wd
        )
        for epoch in range(ne):
            sentence_transformer.train()
            train_loss, batch_orig_embs, batch_cond_embs = 0, [], []
            for i, idx in enumerate(
                tqdm(
                    random.sample(range(len(train_data)), len(train_data)),
                    desc=f"Epoch {epoch + 1}/{ne} - Training",
                )
            ):
                batch_orig_embs.append(
                    sentence_transformer(train_data[idx]["gt_reason_hid"].to(device))
                )
                batch_cond_embs.append(
                    sentence_transformer(train_data[idx]["cond_reason_hid"].to(device))
                )
                if ((i + 1) % args.batch_size == 0) or i == len(train_data) - 1:
                    optimizer.zero_grad()
                    batch_loss = contrastive_loss(
                        torch.cat(batch_orig_embs, dim=0),
                        torch.cat(batch_cond_embs, dim=0),
                    )
                    batch_loss.backward()
                    optimizer.step()
                    train_loss += batch_loss.item()
                    del batch_orig_embs, batch_cond_embs, batch_loss
                    batch_orig_embs, batch_cond_embs = [], []
            avg_train_loss = train_loss / len(train_data)
            # Validation phase
            sentence_transformer.eval()

            with torch.no_grad():
                orig_embs, cond_embs = [], []
                for i, sample in enumerate(
                    tqdm(val_data, desc=f"Epoch {epoch + 1}/{ne} - Validation")
                ):
                    orig_embs.append(
                        sentence_transformer(sample["gt_reason_hid"].to(device))
                    )
                    cond_embs.append(
                        sentence_transformer(sample["cond_reason_hid"].to(device))
                    )
                avg_val_loss = contrastive_loss(
                    torch.cat(orig_embs, dim=0), torch.cat(cond_embs, dim=0)
                ).item() / len(val_data)
                del orig_embs, cond_embs

            # Log metrics
            logger.log_metrics(
                {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, epoch
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(model_path, exist_ok=True)
                sentence_transformer.save_pretrained(model_path)
                logger.logger.info(
                    f"Saved best model with validation loss: {best_val_loss}"
                )
        if ne > 0:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            sentence_transformer = sentence_transformer.cpu()
            del sentence_transformer
            torch.cuda.empty_cache()
            print(model_path)
            sentence_transformer = CustomST.from_pretrained(model_path).to(device)
            logger.logger.info(f"Loading best validation loss = {best_val_loss}")
        for param in sentence_transformer.parameters():
            param.requires_grad = True
        del param
    sentence_transformer = sentence_transformer.cpu()
    del sentence_transformer
    torch.cuda.empty_cache()
    return model_path


def train_contemp_gen(custom_st, train_data, eval_data, logger, args):
    logger.logger.info(f"Training contemp_gen with variation: {args.variation}")
    model_path = f"{args.model_save_path}/contempgen_llr={args.cg_linear_lr}_lwd={args.cg_linear_wd}_le={args.cg_linear_epochs}_llmlr={args.cg_llm_lr}_llmwd={args.cg_llm_wd}_llme={args.cg_llm_epochs}_alpha={args.alpha}_traintok={args.train_max_contemp_tokens}_stargs={args.st_linear_lr}_{args.st_linear_wd}_{args.st_linear_epochs}_{args.st_llm_lr}_{args.st_llm_wd}_{args.st_llm_epochs}"
    if args.variation == "param_analysis":
        model_path = model_path.replace("param_analysis", "vanilla")
        if os.path.exists(model_path):
            return model_path
    
    model_name = args.student_model_name
    if args.variation == "no_small_contemp_gen":
        model_name = args.teacher_model_name
    contemp_gen = ContempGen(
        model_name,
        args.teacher_hid_dim,
        args.variation,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    ).to(args.device)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_name).to(
        args.device
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    del param
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model_name)
    teacher_tok.pad_token = teacher_tok.eos_token

    if args.variation == "no_l_reason":
        args.alpha = 0  # maximize ans loss

    hyper_params = [
        (args.cg_linear_lr, args.cg_linear_wd, args.cg_linear_epochs),
        (args.cg_llm_lr, args.cg_llm_wd, args.cg_llm_epochs),
    ]
    if args.variation == "no_small_contemp_gen":
        hyper_params = [hyper_params[1]]  # no linear layer
    else:  # freeze student model first
        for param in contemp_gen.model.parameters():
            param.requires_grad = False
        del param
    for lr, wd, ne in hyper_params:
        best_val_loss = float("inf")
        optimizer = optim.AdamW(contemp_gen.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(ne):
            contemp_gen.train()
            total_loss, reason_loss, ans_loss, batch_loss = 0, 0, 0, 0
            for i, idx in enumerate(tqdm(
                random.sample(range(len(train_data)), len(train_data)),
                desc=f"Epoch {epoch + 1}/{ne}",
            )):
                l_reason, contemp_states = compute_reason_loss(
                    train_data[idx], contemp_gen, custom_st, args
                )
                l_ans = compute_ans_loss(
                    train_data[idx], teacher_tok, teacher_model, args, contemp_states
                )
                batch_loss += args.alpha * l_reason + (1 - args.alpha) * l_ans
                if ((i + 1) % args.batch_size == 0) or i == len(train_data) - 1:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() / len(train_data)
                    del batch_loss
                    batch_loss = 0
                reason_loss += l_reason.item() / len(train_data)
                ans_loss += l_ans.item() / len(train_data)
                del l_reason, contemp_states, l_ans
                torch.cuda.empty_cache()
            logger.log_metrics(
                {
                    "total_loss": total_loss,
                    "reason_loss": reason_loss,
                    "ans_loss": ans_loss,
                },
                epoch,
            )
            contemp_gen.eval()
            with torch.no_grad():
                eval_loss, eval_r_loss, eval_a_loss = 0, 0, 0
                for idx in tqdm(range(len(eval_data)), desc=f"Epoch {epoch + 1}/{ne}"):
                    l_reason, contemp_states = compute_reason_loss(
                        eval_data[idx], contemp_gen, custom_st, args
                    )
                    l_ans = compute_ans_loss(
                        eval_data[idx], teacher_tok, teacher_model, args, contemp_states
                    )
                    loss = args.alpha * l_reason + (1 - args.alpha) * l_ans
                    eval_loss += loss.item() / len(eval_data)
                    eval_r_loss += l_reason.item() / len(eval_data)
                    eval_a_loss += l_ans.item() / len(eval_data)
                    del l_reason, contemp_states, l_ans, loss
                    torch.cuda.empty_cache()

            logger.log_metrics(
                {
                    "eval_total_loss": eval_loss,
                    "eval_reason_loss": eval_r_loss,
                    "eval_ans_loss": eval_a_loss,
                },
                epoch,
            )

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                os.makedirs(model_path, exist_ok=True)
                contemp_gen.save_pretrained(model_path)
                logger.logger.info(
                    f"Saved best model with validation loss: {best_val_loss}"
                )
        if ne > 0:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            contemp_gen = contemp_gen.cpu()
            del contemp_gen
            gc.collect()
            torch.cuda.empty_cache()
            print(model_path)
            contemp_gen = ContempGen.from_pretrained(model_path).to(args.device)
            logger.logger.info(f"Loading best validation loss = {best_val_loss}")
        if args.variation != "no_small_contemp_gen":  # unfreeze student model
            for param in contemp_gen.model.parameters():
                param.requires_grad = True
            del param
    teacher_model = teacher_model.cpu()
    contemp_gen = contemp_gen.cpu()
    del teacher_model, contemp_gen
    gc.collect()
    torch.cuda.empty_cache()
    return model_path


def get_tokens(item, tok, config, device, max_len):
    query_prompt, ans_prompt = get_prompts(config)
    query = tok(
        query_prompt + item["query"],
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    ans_prompt = tok(
        ans_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    ans = tok(
        item["answer"],
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    return query, ans_prompt, ans


def compute_reason_loss(item, contemp_gen, sentence_transformer, args):
    num_tokens = args.train_max_contemp_tokens
    query, ans_prompt, _ = get_tokens(
        item,
        contemp_gen.tokenizer,
        args.config,
        args.device,
        args.max_seq_len,
    )
    contemp_tokens = (
        torch.tensor([[contemp_gen.thought_token_id] * num_tokens]).long().to(args.device)
    )
    prefix_len = query["input_ids"].shape[-1] - 1
    contemp_inputs = torch.cat(
        [query["input_ids"], contemp_tokens], dim=-1
    )
    contemp_states = contemp_gen(contemp_inputs)
    contemp_states = contemp_states[:, prefix_len : prefix_len + num_tokens]

    gt_reason = item["gt_reason_hid"].to(args.device)
    if args.variation == "no_l_reason":
        similarity = torch.tensor(1).to(args.device) # for l_reason to equal 0
    elif sentence_transformer is None:
        similarity = F.cosine_similarity(
            contemp_states.mean(dim=1).squeeze(),
            gt_reason.mean(dim=1).squeeze(),
            dim=-1,
        )
    else:
        contemp_embeddings = sentence_transformer(contemp_states)
        similarity = F.cosine_similarity(gt_reason, contemp_embeddings, dim=-1)  # FLAG
        del contemp_embeddings
    l_reason = 1 - similarity
    del query, contemp_tokens, ans_prompt, contemp_inputs, similarity, gt_reason
    return l_reason, contemp_states


def compute_ans_loss(item, teacher_tok, teacher_model, args, contemp_states):
    query, ans_prompt, ans = get_tokens(
        item, teacher_tok, args.config, args.device, args.max_seq_len
    )
    # Get the embeddings from the model's embedding layer (no gradients needed)
    with torch.no_grad():
        query_embs = teacher_model.get_input_embeddings()(query["input_ids"])
        ans_prompt_embs = teacher_model.get_input_embeddings()(ans_prompt["input_ids"])
        ans_embs = teacher_model.get_input_embeddings()(ans["input_ids"])

    # Create a new inputs_embeds by concatenating
    new_embs = torch.cat([query_embs, contemp_states, ans_prompt_embs, ans_embs], dim=1)
    batch_size, seq_len, _ = new_embs.shape
    attn_mask = torch.ones((batch_size, seq_len)).long().to(args.device)
    pos_ids = torch.arange(seq_len).long().to(args.device).expand(new_embs.shape[0], -1)
    start_idx = new_embs.shape[1] - ans_embs.shape[1]
    labels = torch.cat(
        [torch.tensor([[-100] * start_idx]).to(args.device), ans["input_ids"]], dim=1
    ).long()

    # Forward pass with combined embeddings
    outputs = teacher_model(
        inputs_embeds=new_embs,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
        labels=labels,
    )
    loss = outputs.loss
    clear_cache_in_dict(outputs)
    del query, ans_prompt, ans, attn_mask, pos_ids, outputs
    del query_embs, ans_prompt_embs, ans_embs, new_embs, labels
    return loss


def run_semcot_inference(logger, contemp_gen, dataset, args):
    num_tokens = args.eval_max_contemp_tokens
    contemp_gen.eval()

    # Load teacher LLM for generating answers
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model_name)
    teacher_tok.pad_token = teacher_tok.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_name).to(
        args.device
    )
    teacher_model.eval()

    all_metrics = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        with torch.no_grad():
            for sample in tqdm(dataset, desc="Running inference"):
                query, ans_prompt, _ = get_tokens(
                    sample,
                    contemp_gen.tokenizer,
                    args.config,
                    args.device,
                    args.max_seq_len,
                )
                contemp_tokens = (
                    torch.tensor([[contemp_gen.thought_token_id] * num_tokens]).long().to(args.device)
                )
                prefix_len = query["input_ids"].shape[-1] - 1
                contemp_inputs = torch.cat(
                    [query["input_ids"], contemp_tokens],
                    dim=-1,
                )
                contemp_start = time.time()
                # Get contemplation states from the correct position
                contemp_states = contemp_gen(contemp_inputs)
                contemp_states = contemp_states[:, prefix_len : prefix_len + num_tokens]
                contemp_time = time.time() - contemp_start

                query, ans_prompt, _ = get_tokens(
                    sample, teacher_tok, args.config, args.device, args.max_seq_len
                )
                query_embs = teacher_model.get_input_embeddings()(query["input_ids"])
                ans_embs = teacher_model.get_input_embeddings()(ans_prompt["input_ids"])
                comb_embs = torch.cat([query_embs, contemp_states, ans_embs], dim=1)
                attn_mask = torch.ones((1, comb_embs.shape[1])).long().to(args.device)

                # Generate answer with the combined embeddings directly
                gen_start = time.time()
                outputs = teacher_model.generate(
                    inputs_embeds=comb_embs,
                    attention_mask=attn_mask,
                    max_length=30 + comb_embs.size(1),
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=teacher_tok.eos_token_id,
                )
                gen_time = time.time() - gen_start

                # Decode only the generated part (skip the prompt and contemplation tokens)
                prefix_length = (
                    comb_embs.shape[1] - 1 if len(outputs[0]) > comb_embs.shape[1] else 0
                )
                answer = teacher_tok.decode(
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
                del query, ans_prompt, contemp_states, contemp_tokens
                del query_embs, ans_embs, comb_embs, outputs, attn_mask
                torch.cuda.empty_cache()
            metrics = {
                "numerical_accuracy": float(np.mean([r["correct"] for r in results])),
                "ave_sample_time": float(np.mean([r["sample_time"] for r in results])),
                "dataset": args.dataset,
                "eval_temp": temp,
                "config": args.config,
                "st_linear_lr": args.st_linear_lr,
                "st_linear_wd": args.st_linear_wd,
                "st_linear_epochs": args.st_linear_epochs,
                "st_llm_lr": args.st_llm_lr,
                "st_llm_wd": args.st_llm_wd,
                "st_llm_epochs": args.st_llm_epochs,
                "cg_linear_lr": args.cg_linear_lr,
                "cg_linear_wd": args.cg_linear_wd,
                "cg_linear_epochs": args.cg_linear_epochs,
                "cg_llm_lr": args.cg_llm_lr,
                "cg_llm_wd": args.cg_llm_wd,
                "cg_llm_epochs": args.cg_llm_epochs,
                "train_max_contemp_tokens": args.train_max_contemp_tokens,
                "eval_max_contemp_tokens": num_tokens,
                "alpha": args.alpha,
                "variation": args.variation,
            }
            logger.logger.info(
                f"eval_temp = {temp} | acc = {metrics['numerical_accuracy']} | ave_sample_time = {metrics['ave_sample_time']}"
            )
            all_metrics.append(metrics)
    return all_metrics
