import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import LoraConfig, TaskType, get_peft_model
from utils.utils import get_prompts, clear_cache_in_dict


class Coconut(nn.Module):
    """
    Implementation of Chain of Continuous Thought (Coconut) model from the paper
    "Training Large Language Models to Reason in a Continuous Latent Space"
    by Shibo Hao et al.
    """

    def __init__(
        self,
        base_model_name,
        r,
        lora_alpha,
        lora_dropout,
        config,
        num_cont_thoughts=2,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.config = config
        self.num_cont_thoughts = num_cont_thoughts

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Create special tokens for Coconut
        self.bot_token = "<bot>"  # Beginning of thought token
        self.thought_token = "<thought>"  # continuous thought token
        self.eot_token = "<eot>"  # End of thought token
        special_tokens = {
            "additional_special_tokens": [
                self.bot_token,
                self.eot_token,
                self.thought_token,
            ]
        }
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

    def process_sample(self, sample, max_seq_len, device, max_tokens, stage=-1):
        bot_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.bot_token)]]
        ).to(device)
        eot_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.eot_token)]]
        ).to(device)
        thought_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.thought_token)]]
        ).to(device)
        query_prompt, ans_prompt = get_prompts(self.config)
        query = self.tokenizer(
            query_prompt + sample["query"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        ).to(device)
        ans_prompt = self.tokenizer(
            ans_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        ).to(device)
        if stage == 0:  # Standard CoT
            reasoning = self.tokenizer(
                sample["reasoning"],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            )["input_ids"].to(device)
            num_tokens = 0
        else:  # Replace first 'stage' steps with continuous thoughts
            steps = [
                step.strip() for step in sample["reasoning"].split("\n") if step.strip()
            ]
            num_tokens = min(stage, len(steps)) * self.num_cont_thoughts
            if stage == -1:
                reasoning = torch.tensor([]).long().to(device)
                num_tokens = min(len(steps) * self.num_cont_thoughts, max_tokens)
            else:
                num_tokens = min(stage, len(steps)) * self.num_cont_thoughts
                reasoning = self.tokenizer(
                    " ".join(steps[min(stage, len(steps)) :]),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                    add_special_tokens=False,
                )["input_ids"].to(device)
                num_tokens = min(num_tokens, max_tokens)
            thought_tokens = torch.tensor([[thought_token_id] * num_tokens]).to(device)
            reasoning = torch.cat([thought_tokens, eot_token_id, reasoning], dim=-1)
            del thought_tokens
        input_ids = torch.cat(
            [query["input_ids"], bot_token_id, reasoning, ans_prompt["input_ids"]],
            dim=-1,
        )
        bot_idx = query["input_ids"].shape[-1]
        eot_idx = query["input_ids"].shape[-1] + num_tokens + 1
        label = None
        if stage != -1:
            ans = self.tokenizer(
                sample["answer"],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            ).to(device)
            start_idx = input_ids.shape[-1]
            input_ids = torch.cat([input_ids, ans["input_ids"]], dim=-1)
            label = input_ids.clone()
            label[:, :start_idx] = -100
            del ans
        attn_mask = torch.ones_like(input_ids).to(device)
        del bot_token_id, eot_token_id, thought_token_id, query, ans_prompt, reasoning
        return input_ids, attn_mask, label, bot_idx, eot_idx

    def forward(self, input_embs, attn_mask, bot_idx, eot_idx, label):
        # Initial forward pass to get the hidden states for the query
        outputs = self.model(
            inputs_embeds=input_embs[:, : bot_idx + 1],
            attention_mask=attn_mask[:, : bot_idx + 1],
            output_hidden_states=True,
        )
        cont_thoughts = []
        if (eot_idx - bot_idx - 1) > 0:
            cont_thoughts = [outputs.hidden_states[-1][:, -1].unsqueeze(1)]
            clear_cache_in_dict(outputs)
            # Autoregressively generate continuous thought tokens
            for i in range(eot_idx - bot_idx - 2):
                inputs = torch.cat(
                    ([input_embs[:, : bot_idx + 1]] + cont_thoughts), dim=1
                ).to(input_embs.device)
                outputs = self.model(
                    inputs_embeds=inputs,
                    attention_mask=attn_mask[:, : bot_idx + 1 + i],
                    output_hidden_states=True,
                )
                cont_thoughts.append(outputs.hidden_states[-1][:, -1].unsqueeze(1))
                clear_cache_in_dict(outputs)
                del inputs
        clear_cache_in_dict(outputs)
        new_inputs = torch.cat(
            (
                [input_embs[:, : bot_idx + 1]]
                + cont_thoughts
                + [input_embs[:, eot_idx:]]
            ),
            dim=1,
        )
        outputs = self.model(
            inputs_embeds=new_inputs,
            attention_mask=attn_mask,
            output_hidden_states=True,
            labels=label,
        )
        del cont_thoughts
        return new_inputs, outputs

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained Coconut model

        Args:
            path: Path to the saved model

        Returns:
            Loaded Coconut model
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
        Save the Coconut model

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
