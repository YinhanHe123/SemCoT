import gc
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import copy
import os
from utils.utils import clear_cache_in_dict
from huggingface_hub import PyTorchModelHubMixin

class CustomST(nn.Module):
    def __init__(
        self, base_model_name, start_layer_idx=16, end_layer_idx=20, embedding_dim=768
    ):
        super().__init__()

        # Load the base model, tokenizer and config
        base_model = AutoModel.from_pretrained(base_model_name)

        # Extract layer indices
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx

        # Store model name for architecture identification
        self.base_model_name = base_model_name

        # Get input dimension from the base model
        self.input_dim = base_model.config.hidden_size

        self.norm = copy.deepcopy(base_model.norm)
        self.rotary_emb = copy.deepcopy(base_model.rotary_emb)
        # Extract the required transformer layers
        self.layers = nn.ModuleList()
        for i in range(start_layer_idx, end_layer_idx + 1):
            if i < len(base_model.layers):
                self.layers.append(copy.deepcopy(base_model.layers[i]))

        # Add embedding projection layer for sentence embeddings
        self.embedding_projection = nn.Linear(self.input_dim, embedding_dim)
        del base_model
        torch.cuda.empty_cache()

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Process hidden states through the extracted layers

        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask for the sequence (batch_size, seq_len)
            position_ids: Optional position IDs (batch_size, seq_len)

        Returns:
            Tensor: Sentence embeddings
        """
        device = hidden_states.device
        batch_size, seq_length, _ = hidden_states.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length).expand(batch_size, -1).to(device)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length)).to(device)

        # Generate position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through the extracted layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        # Apply normalization
        hidden_states = self.norm(hidden_states)

        # Apply attention mask for proper mean pooling
        if attention_mask is not None:
            if attention_mask.dim() < hidden_states.dim():
                attention_mask = attention_mask.unsqueeze(-1).expand(
                    hidden_states.size()
                )
            sum_mask = torch.clamp(
                attention_mask.sum(dim=1), min=1e-9
            )  # Avoid division by zero
            pooled_output = (hidden_states * attention_mask).sum(dim=1) / sum_mask
            del sum_mask
        else:
            # Simple mean pooling
            pooled_output = hidden_states.mean(dim=1)

        # Project to embedding space
        sentence_embedding = self.embedding_projection(pooled_output)
        del pooled_output, hidden_states, attention_mask, position_ids, position_embeddings
        return sentence_embedding

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pre-trained sentence transformer
        """
        # Load configuration
        config = torch.load(os.path.join(path, "config.pt"))

        # Initialize model
        model = cls(config["base_model_name"])

        # Load model weights
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        return model

    def save_pretrained(self, path):
        """
        Save the model to disk
        """
        os.makedirs(path, exist_ok=True)

        # Save configuration
        config = {
            "base_model_name": self.base_model_name,
        }
        torch.save(config, os.path.join(path, "config.pt"))
        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))


class ContempGen(nn.Module):
    def __init__(
        self, model_name, teacher_hid_dim, variation, r, lora_alpha, lora_dropout, contemp_gen_name=None
    ):
        super().__init__()
        self.model_name = model_name
        self.variation = variation
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.teacher_hid_dim = teacher_hid_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.thought_token = "<thought>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = {
            "additional_special_tokens": [self.thought_token]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        # Resize token embeddings if new tokens were added

        self.thought_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.thought_token)]]
        )

        def freeze_old_weights_hook(grad):
            return torch.nan_to_num(grad) * torch.concat(
                [torch.zeros_like(grad[:-1]), torch.ones_like(grad[-1:])], dim=0
            ).to(grad.device)

        # Choose which model to use based on variation
        if variation == "no_small_contemp_gen":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
            # Use teacher model with LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=r,  # rank
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],  # Target attention layers
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.model.model.embed_tokens.weight.requires_grad = True
            self.model.model.lm_head.weight.requires_grad = True
            self.model.model.lm_head.weight.register_hook(freeze_old_weights_hook)
            self.model.model.model.embed_tokens.weight.register_hook(
                freeze_old_weights_hook
            )
            # No projection needed as we're already using the teacher dimensions
            self.projection_layer = nn.Identity()
        else:
            if contemp_gen_name is not None:
                self.model = SemoCoT.from_pretrained(contemp_gen_name)
                self.projection_layer = self.model.projection_layer
            else:
                self.model = AutoModel.from_pretrained(model_name)
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                self.projection_layer = nn.Linear(
                    self.model.config.hidden_size, teacher_hid_dim
                )
            self.model.embed_tokens.weight.requires_grad = True
            self.model.embed_tokens.weight.register_hook(
                freeze_old_weights_hook
            )
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
        if len(input_ids.shape) == 3:
            batch_size, seq_length, _ = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length)).to(device)

        # Generate model hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Project to teacher model hidden dimension
        projected_states = self.projection_layer(outputs.hidden_states[-1])
        clear_cache_in_dict(outputs)
        del outputs, attention_mask
        gc.collect()
        return projected_states

    @classmethod
    def from_pretrained(cls, path):
        # Load the model config from the saved path
        config_dict = torch.load(os.path.join(path, "config.pt"))

        # Initialize the model with the loaded config
        model = cls(
            config_dict["model_name"],
            config_dict["teacher_hid_dim"],
            config_dict["variation"],
            config_dict["r"],
            config_dict["lora_alpha"],
            config_dict["lora_dropout"],
        )

        # Load the state dict
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        return model

    def save_pretrained(self, path):
        # Save model config
        config_dict = {
            "model_name": self.model_name,
            "teacher_hid_dim": self.teacher_hid_dim,
            "variation": self.variation,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }
        os.makedirs(path, exist_ok=True)
        torch.save(config_dict, os.path.join(path, "config.pt"))

        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

class SemCoT(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="text-generation",
):
    def __init__(self, model_name, teacher_hid_dim):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(self.model.embed_tokens.num_embeddings + 1)
        self.projection_layer = nn.Linear(self.model.config.hidden_size, teacher_hid_dim)

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
        if len(input_ids.shape) == 3:
            batch_size, seq_length, _ = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length)).to(device)

        # Generate model hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Project to teacher model hidden dimension
        projected_states = self.projection_layer(outputs.hidden_states[-1])
        clear_cache_in_dict(outputs)
        del outputs, attention_mask
        gc.collect()
        return projected_states