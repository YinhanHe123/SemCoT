import os
import random
import numpy as np
import torch
import json
import transformers
import re

BEST_PARAMS = {'small': {'coin_flip': {'stllr': 0.001,
   'stlwd': 0.0001,
   'stllmlr': 0.001,
   'stllmwd': 0.01,
   'stle': 5,
   'stllme': 2,
   'cgllr': 0.0001,
   'cglwd': 0.0001,
   'cgllmlr': 0.001,
   'cgllmwd': 1e-07,
   'cgle': 5,
   'cgllme': 1,
   'alpha': 0.5},
  'commonsense_qa': {'stllr': 0.001,
   'stlwd': 0.001,
   'stllmlr': 0.001,
   'stllmwd': 1e-07,
   'stle': 5,
   'stllme': 3,
   'cgllr': 0.01,
   'cglwd': 0.01,
   'cgllmlr': 0.001,
   'cgllmwd': 0.001,
   'cgle': 5,
   'cgllme': 3,
   'alpha': 0.75},
  'gsm8k': {'stllr': 0.001,
   'stlwd': 0.01,
   'stllmlr': 1e-07,
   'stllmwd': 0.01,
   'stle': 7,
   'stllme': 1,
   'cgllr': 0.001,
   'cglwd': 0.001,
   'cgllmlr': 0.001,
   'cgllmwd': 1e-07,
   'cgle': 7,
   'cgllme': 2,
   'alpha': 0.25},
  'multiarith': {'stllr': 0.01,
   'stlwd': 0.0001,
   'stllmlr': 0.001,
   'stllmwd': 1e-05,
   'stle': 7,
   'stllme': 1,
   'cgllr': 0.001,
   'cglwd': 0.01,
   'cgllmlr': 1e-07,
   'cgllmwd': 1e-05,
   'cgle': 5,
   'cgllme': 1,
   'alpha': 0.75},
  'svamp': {'stllr': 0.01,
   'stlwd': 0.001,
   'stllmlr': 0.001,
   'stllmwd': 0.01,
   'stle': 5,
   'stllme': 1,
   'cgllr': 0.001,
   'cglwd': 0.01,
   'cgllmlr': 0.001,
   'cgllmwd': 0.01,
   'cgle': 5,
   'cgllme': 2,
   'alpha': 0.25}},
 'mistral': {'coin_flip': {'stllr': 0.0001,
   'stlwd': 0.01,
   'stllmlr': 1e-07,
   'stllmwd': 1e-05,
   'stle': 7,
   'stllme': 3,
   'cgllr': 0.001,
   'cglwd': 0.0001,
   'cgllmlr': 1e-07,
   'cgllmwd': 1e-07,
   'cgle': 3,
   'cgllme': 2,
   'alpha': 0.5},
  'commonsense_qa': {'stllr': 0.0001,
   'stlwd': 0.0001,
   'stllmlr': 1e-05,
   'stllmwd': 1e-07,
   'stle': 3,
   'stllme': 2,
   'cgllr': 0.001,
   'cglwd': 0.001,
   'cgllmlr': 0.001,
   'cgllmwd': 0.001,
   'cgle': 1,
   'cgllme': 3,
   'alpha': 0.25},
  'gsm8k': {'stllr': 0.001,
   'stlwd': 0.001,
   'stllmlr': 1e-05,
   'stllmwd': 1e-05,
   'stle': 5,
   'stllme': 1,
   'cgllr': 0.001,
   'cglwd': 0.01,
   'cgllmlr': 1e-05,
   'cgllmwd': 1e-05,
   'cgle': 1,
   'cgllme': 2,
   'alpha': 0.25},
  'multiarith': {'stllr': 0.0001,
   'stlwd': 0.01,
   'stllmlr': 1e-07,
   'stllmwd': 0.001,
   'stle': 5,
   'stllme': 2,
   'cgllr': 0.0001,
   'cglwd': 0.01,
   'cgllmlr': 1e-05,
   'cgllmwd': 1e-05,
   'cgle': 1,
   'cgllme': 2,
   'alpha': 0.25},
  'svamp': {'stllr': 0.01,
   'stlwd': 0.0001,
   'stllmlr': 1e-07,
   'stllmwd': 1e-07,
   'stle': 1,
   'stllme': 2,
   'cgllr': 0.001,
   'cglwd': 0.0001,
   'cgllmlr': 1e-05,
   'cgllmwd': 1e-05,
   'cgle': 1,
   'cgllme': 2,
   'alpha': 0.25}}}

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data, filepath):
    """Save data as a JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_directory(file_path):
    """Create directory if it doesn't exist"""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path) if not os.path.exists(dir_path) else None
    return


def append_to_jsonl_file(file_path, new_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Append the new data as a separate line
    with open(file_path, "a") as f:
        f.write(json.dumps(new_data) + "\n")


def evaluate_pred(pred, gt, data_name):
    if data_name == "commonsense_qa":
        pred = pred.strip().lower()
        gt = gt.strip().lower()
        return any(letter == gt for letter in pred)
    elif data_name == "coin_flip":
        possible_answers = {
            "yes": ["yes", "heads", "still heads"],
            "no": ["no", "tails", "not heads"],
        }
        pred = pred.strip().lower().split()
        gt = gt.strip().lower()
        return any(pos_ans in pred for pos_ans in possible_answers[gt])
    else:
        numbers = re.findall(r"-?\d+\.?\d*", pred)
        pred_nums = [float(num) for num in numbers]
        gt = float(gt.strip().replace(",", ""))
        return any(abs(pred - gt) < 1e-6 for pred in pred_nums)


def get_prompts(config):
    if config == "mistral":
        query_prompt = "<s>[INST] Question: "
        ans_prompt = "\n Answer: [/INST]"
    elif config == "small":
        query_prompt = "Question: "
        ans_prompt = "\n Answer:"
    elif config == "qwen":
        query_prompt = "<|im_start|>user\nQuestion: "
        ans_prompt = "<|im_end|>\n<|im_start|>assistant\nAnswer: "
    return query_prompt, ans_prompt


def clear_cache_in_dict(dict_to_clear):
    for k in dict_to_clear:
        dict_to_clear[k] = None
