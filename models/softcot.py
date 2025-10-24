import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import os
from utils.utils import clear_cache_in_dict, get_prompts


class SoftCoT(nn.Module):
    """
    Implementation of Soft Chain of Thought model from the paper
    "SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs"
    by Yige Xu, Xu Guo, Zhiwei Zeng, Chunyan Miao.
    """

    def __init__(self, config, llm_model_name, assist_model_name):
        """
        Initialize the SoftCoT model.

        Args:
            llm_model_name: Name of the backbone LLM
            assistant_model_name: Name of the assistant model to generate soft thought tokens
        """
        super().__init__()
        self.config = config
        self.llm_model_name = llm_model_name
        self.assist_model_name = assist_model_name

        # Load the LLM model and tokenizer
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm_model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        for p in self.llm_model.parameters():
            p.requires_grad = False

        # Load the assistant model and tokenizer
        self.assist_model = AutoModel.from_pretrained(assist_model_name)
        self.assist_model.eval()
        self.assist_tok = AutoTokenizer.from_pretrained(assist_model_name)
        self.assist_tok.pad_token = self.assist_tok.eos_token
        for p in self.assist_model.parameters():
            p.requires_grad = False

        # Initialize the projection module to map assistant model's hidden states to LLM's space
        assist_hid_dim = self.assist_model.config.hidden_size
        llm_hid_dim = self.llm_model.config.hidden_size
        self.proj = nn.Linear(assist_hid_dim, llm_hid_dim)

    def generate_soft_thoughts(self, query, num_tokens, max_seq_len, device):
        """
        Generate soft thought tokens based on the query using the assistant model.

        Args:
            query: The input query/question
            num_tokens: Number of soft thought tokens to generate

        Returns:
            Soft thought tokens (hidden states from the assistant model)
        """
        # Prepare input with [UNK] tokens as placeholders for soft thoughts
        instruction = "You are helping a large language model solve reasoning problems. Generate helpful thoughts."
        unk_toks = [[self.assist_tok.unk_token_id] * num_tokens]
        if not self.assist_tok.unk_token:
            unk_toks = [[self.assist_tok.eos_token_id] * num_tokens]

        query_prompt, _ = get_prompts(self.config)

        # Construct the input
        inputs = self.assist_tok(
            f"{instruction}\n{query_prompt} {query}\n",
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        ).to(device)
        prefix_len = inputs["input_ids"].shape[-1]
        inputs = torch.cat(
            [inputs["input_ids"], torch.tensor(unk_toks).to(device)], dim=-1
        )

        # Generate hidden states with the assistant model
        with torch.no_grad():
            outputs = self.assist_model(input_ids=inputs, output_hidden_states=True)
            soft_thought_tokens = outputs.hidden_states[-1][:, prefix_len:]
        soft_thought_tokens = self.proj(soft_thought_tokens)
        clear_cache_in_dict(outputs)
        del outputs, inputs
        return soft_thought_tokens

    def get_combined_inputs(
        self, thought_toks, sample, max_seq_len, device, append_ans=True
    ):
        """
        Project the soft thought tokens to LLM's space and create LLM inputs
        """

        query_prompt, ans_prompt = get_prompts(self.config)
        query = self.llm_tokenizer(
            query_prompt + sample["query"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        ans_prompt = self.llm_tokenizer(
            ans_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        ans = self.llm_tokenizer(
            sample["answer"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        query_embs = self.llm_model.get_input_embeddings()(query)
        ans_prompt_embs = self.llm_model.get_input_embeddings()(ans_prompt)
        comb_embs = torch.cat([query_embs, thought_toks, ans_prompt_embs], dim=1)
        label = torch.ones(1, comb_embs.shape[1]).long().to(device) * -100
        if append_ans:
            ans_embs = self.llm_model.get_input_embeddings()(ans)
            comb_embs = torch.cat([comb_embs, ans_embs], dim=1)
            label = torch.cat([label, ans], dim=1)
        return comb_embs, label

    def save_pretrained(self, path):
        """
        Save the SoftCoT model (specifically the projection module) to disk.

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save the projection module
        torch.save(self.proj.state_dict(), os.path.join(path, "proj_module.pt"))

        # Save the config
        config = {
            "config": self.config,
            "llm_model_name": self.llm_model_name,
            "assist_model_name": self.assist_model_name,
        }
        torch.save(config, os.path.join(path, "config.pt"))

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained SoftCoT model.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded SoftCoT model
        """
        # Load config
        config = torch.load(os.path.join(path, "config.pt"))

        # Initialize model with loaded config
        model = cls(
            config["config"],
            config["llm_model_name"],
            config["assist_model_name"],
        )

        # Initialize and load the projection module
        model.proj.load_state_dict(
            torch.load(os.path.join(path, "proj_module.pt"), map_location="cpu")
        )
        return model
