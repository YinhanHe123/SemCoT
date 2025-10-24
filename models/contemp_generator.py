import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os

class ContemplationGenerator(nn.Module):
    def __init__(self, model_name, teacher_hid_dim, variation, r, lora_alpha, lora_dropout):
        super().__init__()
        self.model_name = model_name
        self.variation = variation
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end of sequence token

        # Choose which model to use based on variation
        if variation == "no_small_contemp_gen":
            # Use teacher model with LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=r,  # rank
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"]  # Target attention layers
            )
            self.model = get_peft_model(self.model, peft_config)            
            # No projection needed as we're already using the teacher dimensions
            self.projection_layer = nn.Identity()
        else:
            # Create projection layer to match dimensions if needed
            self.projection_layer = nn.Linear(self.model.config.hidden_size, teacher_hid_dim)
        self.teacher_hid_dim = teacher_hid_dim

    def forward(self, input_ids, attention_mask=None):
        # Generate model hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Project to teacher model hidden dimension
        projected_states = self.projection_layer(outputs.hidden_states[-1])
        del outputs
        return projected_states

    @classmethod
    def from_pretrained(cls, path):
        # Load the model config from the saved path
        config_dict = torch.load(f"{path}/config.pt")

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
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location='cpu'))
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
        torch.save(config_dict, f"{path}/config.pt")

        # Save model weights
        torch.save(self.state_dict(), f"{path}/model.pt")