import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
from utils.utils import clear_cache_in_dict, get_prompts


class CODI(nn.Module):
    """
    Implementation of Continuous Chain-of-Thought via Self-Distillation (CODI) model
    from the paper "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation"
    by Zhenyi Shen et al.
    """

    def __init__(self, base_model_name, r, lora_alpha, lora_dropout, config):
        """
        Initialize the CODI model.
        """
        super().__init__()
        self.base_model_name = base_model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.config = config

        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens for CODI
        self.bot_token = "<bot>"  # Beginning of thought token
        self.eot_token = "<eot>"  # End of thought token
        special_tokens = {"additional_special_tokens": [self.bot_token, self.eot_token]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        # Resize token embeddings if new tokens were added
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Projection layer for continuous thoughts
        self.projection_layer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
        )

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

    def process_sample(self, sample, max_seq_len, device):
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
        return query, ans_prompt, cot, ans

    def forward_student(self, query, ans_prompt, num_tokens, ans=None):
        """
        Student forward pass: Generate continuous thoughts autoregressively

        """
        bot_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.bot_token)]]
        ).to(query.device)
        eot_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.eot_token)]]
        ).to(query.device)
        begin_inputs = torch.cat([query, bot_token_id], dim=1)
        student_outputs = self.model(input_ids=begin_inputs, output_hidden_states=True)
        continuous_tokens = []
        latent = self.projection_layer(
            student_outputs.hidden_states[-1][:, -1].unsqueeze(1)
        )
        continuous_tokens.append(latent)
        past_key_values = student_outputs.past_key_values
        clear_cache_in_dict(student_outputs)
        for _ in range(num_tokens - 1):
            student_outputs = self.model(
                inputs_embeds=latent,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            latent = self.projection_layer(student_outputs.hidden_states[-1])
            continuous_tokens.append(latent)
            past_key_values = student_outputs.past_key_values
            clear_cache_in_dict(student_outputs)
        continuous_tokens = torch.cat(continuous_tokens, dim=1)
        end_inputs = torch.cat([eot_token_id, ans_prompt], dim=-1)
        labels, start_idx = end_inputs.clone(), end_inputs.shape[-1]
        if ans is not None:
            end_inputs = torch.cat([end_inputs, ans], dim=-1)
            labels = end_inputs.clone()
            labels[:, :start_idx] = -100
        student_outputs = self.model(
            input_ids=end_inputs,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
            labels=labels,
        )
        del bot_token_id, eot_token_id, labels, latent, past_key_values
        return student_outputs, continuous_tokens, start_idx, begin_inputs, end_inputs

    def forward_teacher(self, query, cot, ans, ans_prompt):
        """
        Teacher forward pass: Process query with the ground-truth CoT
        """
        inputs = torch.cat([query, cot, ans_prompt, ans], dim=-1)
        labels = inputs.clone()
        start_idx = query.shape[1] + cot.shape[1] + ans_prompt.shape[1]
        labels[:, :query.shape[1]] = -100
        teacher_outputs = self.model(
            input_ids=inputs, output_hidden_states=True, labels=labels
        )
        del inputs, labels
        return teacher_outputs, start_idx

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained CODI model

        Args:
            path: Path to the saved model

        Returns:
            Loaded CODI model
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
        Save the CODI model

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
