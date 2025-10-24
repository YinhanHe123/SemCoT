import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import LoraConfig, TaskType, get_peft_model
from utils.utils import get_prompts


class Pause(nn.Module):
    """
    Implementation of the 'Think Before You Speak' baseline with pause tokens
    Based on: Goyal et al. "Think Before You Speak: Training Language Models With Pause Tokens"
    """

    def __init__(
        self,
        base_model_name,
        r,
        lora_alpha,
        lora_dropout,
        config,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.config = config

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Create special tokens for Pause
        self.pause_token = "<pause>"
        special_tokens = {"additional_special_tokens": [self.pause_token]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        # Resize token embeddings if new tokens were added
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

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
        self.model.model.lm_head.weight.requires_grad = True
        self.model.model.model.embed_tokens.weight.requires_grad = True

        def freeze_old_weights_hook(grad):
            return torch.nan_to_num(grad) * torch.concat(
                [torch.zeros_like(grad[:-1]), torch.ones_like(grad[-1:])], dim=0
            ).to(grad.device)

        self.model.model.lm_head.weight.register_hook(freeze_old_weights_hook)
        self.model.model.model.embed_tokens.weight.register_hook(
            freeze_old_weights_hook
        )

    def process_sample(self, sample, max_seq_len, device, max_tokens, append_ans=True):
        pause_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.pause_token)]]
        ).to(device)
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
        pause_tokens = torch.tensor([[pause_token_id] * max_tokens]).to(device)
        input_ids = torch.cat([query, pause_tokens, ans_prompt], dim=-1)
        attn_mask = torch.ones_like(input_ids).to(device)

        ans = self.tokenizer(
            sample["answer"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        label = torch.cat([input_ids, ans], dim=-1)
        label[:, : input_ids.shape[-1]] = -100
        if append_ans:
            input_ids = torch.cat([input_ids, ans], dim=-1)
            attn_mask = torch.ones_like(input_ids).to(device)
        return input_ids, attn_mask, label

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained Pause model

        Args:
            path: Path to the saved model

        Returns:
            Loaded Pause model
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
        Save the Pause model

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
