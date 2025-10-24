import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, LogitsProcessor, StoppingCriteriaList, LogitsProcessorList
import os
from peft import LoraConfig, TaskType, get_peft_model
from utils.utils import get_prompts

def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    positions = torch.arange(truncate_length)
    lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
    cum_prob = lambda_distribution.sum()
    lambda_distribution[-1] += (1-cum_prob)
    return lambda_distribution

class ICoT_SI(nn.Module):
    """
    Implicit Chain of Thought Stepwise Internalization baseline
    Based on: "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"
    """

    def __init__(
        self,
        base_model_name,
        r,
        lora_alpha,
        lora_dropout,
        config,
        lamdba=4
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.config = config
        self.lambda_distribution = compute_lambda_distribution(lamdba)

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
        self.stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,  # rank
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(self.model, peft_config)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def process_sample(
        self, sample, max_seq_len, device, tok_to_remove=-1, append_ans=True
    ):
        """Prepare a batch for training with CoT removal strategy"""
        query_prompt, ans_prompt = get_prompts(self.config)
        query = self.tokenizer(
            query_prompt + sample["query"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        ans_prompt = self.tokenizer(
            ans_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        cot = self.tokenizer(
            sample["reasoning"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        ans = self.tokenizer(
            sample["answer"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        eot_tok = torch.tensor([[self.tokenizer.eos_token_id]]).to(device)
        input_ids = torch.cat([query, eot_tok, cot, eot_tok, ans_prompt], dim=-1)
        if append_ans:
            input_ids = torch.cat([input_ids, ans], dim=-1)
        labels = input_ids.clone()
        eot_idx1 = query.shape[-1]
        eot_idx2 = query.shape[-1] + cot.shape[-1] + 1
        labels[0, : eot_idx2 + 1] = -100
        pos_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(input_ids)
        masked_complete_cot = False

        # Apply CoT removal if scheduled
        if tok_to_remove == -1:
            tok_to_remove = eot_idx2 - eot_idx1 + 1
        tok_to_remove += torch.multinomial(self.lambda_distribution, 1, replacement=True).item()
        r_from = eot_idx1 + 1
        r_to = min(r_from + tok_to_remove, eot_idx2)
        masked_complete_cot = r_to == eot_idx2
        pos_ids[:, r_from:] += r_to - r_from

        # Create new sequences with removed tokens
        input_ids = torch.cat([input_ids[:, :r_from], input_ids[:, r_to:]], dim=-1)
        pos_ids = pos_ids[:, : input_ids.shape[1]]
        attn_mask = torch.ones_like(input_ids)
        labels = torch.cat([labels[:, :r_from], labels[:, r_to:]], dim=-1)
        return input_ids, attn_mask, pos_ids, labels, masked_complete_cot

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained ICoT_SI model

        Args:
            path: Path to the saved model

        Returns:
            Loaded ICoT_SI model
        """
        # Load config
        config = torch.load(os.path.join(path, "config.pt"))

        # Initialize model with loaded config
        model = cls(
            config["base_model_name"],
            config["r"],
            config["lora_alpha"],
            config["lora_dropout"],
            config["config"],
        )

        # Load state dict
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        return model

    def save_pretrained(self, path):
        """
        Save the ICoT_SI model

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "base_model_name": self.base_model_name,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "config": self.config,
        }
        torch.save(config, os.path.join(path, "config.pt"))

        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id_ = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id_ = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id_] = 0
        return scores