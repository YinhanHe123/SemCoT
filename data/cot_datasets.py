import json
import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.gpt4pair import ReasoningPairsGenerator

DATASET_MAP = {
    "gsm8k": ["openai/gsm8k", "main", "test"],
    "svamp": ["ChilleD/SVAMP", "default", "test"],
    "multiarith": ["ChilleD/MultiArith", "default", "test"],
    "commonsense_qa": ["tau/commonsense_qa", None, "validation"],
    "coin_flip": ["skrishna/coin_flip", None, "test"],
}


def preprocess_data(dataset_name, data_path, val_name, config_name, train_path, eval_path):
    new_data = [[], []]
    for i, path in enumerate([train_path, eval_path]):
        if os.path.exists(path):
            new_data[i] = json.load(open(path, "r", encoding="utf-8"))
    generator = ReasoningPairsGenerator()
    raw = load_dataset(data_path, config_name)
    train_data = raw["train"].select(range(min(800, len(raw["train"]))))
    eval_data = raw[val_name].select(range(min(200, len(raw[val_name]))))
    if len(new_data[0]) == 0:
        new_data[0] = [d for d in train_data]
    if len(new_data[1]) == 0:
        new_data[1] = [d for d in eval_data]
    for i, (data, path) in enumerate(
        [(new_data[0], train_path), (new_data[1], eval_path)]
    ):
        for j, sample in enumerate(tqdm(data)):
            if all(
                key in new_data[i][j]
                for key in ["query", "full_answer", "reasoning", "answer"]
            ):
                continue
            if "gsm8k" in dataset_name:
                new_data[i][j] = {
                    "query": sample["question"],
                    "full_answer": sample["answer"],
                    "reasoning": sample["answer"].split("####")[0].strip(),
                    "answer": sample["answer"].split("####")[1].strip(),
                }
            elif "svamp" in dataset_name:
                new_data[i][j] = {
                    "query": sample["question"],
                    "full_answer": sample["answer"],
                    "reasoning": sample["answer"].split("####")[0].strip(),
                    "answer": sample["answer"].split("####")[1].strip(),
                }
            elif "multiarith" in dataset_name:
                reasoning = generator.generate_reasoning(
                    sample["question"], dataset_name
                )
                if "####" in reasoning:
                    reasoning, answer = reasoning.split("####")
                    reasoning, answer = reasoning.strip(), answer.strip()
                    if len(answer) == 0:
                        answer = sample["answerKey"].strip()
                else:
                    reasoning, answer = reasoning.strip(), sample["final_ans"].strip()
                new_data[i][j] = {
                    "query": sample["question"],
                    "full_answer": reasoning + "\n####" + answer,
                    "reasoning": reasoning,
                    "answer": answer,
                }
            elif "commonsense_qa" in dataset_name:
                choices = [
                    (label, text)
                    for label, text in zip(*list(sample["choices"].values()))
                ]
                choices = "\n".join([f"{c[0]}. {c[1]}" for c in choices])
                question = f"{sample['question']}\nChoices:\n{choices}\n"
                reasoning = generator.generate_reasoning(question, dataset_name)
                if "####" in reasoning:
                    reasoning, answer = reasoning.split("####")
                    reasoning, answer = reasoning.strip(), answer.strip()
                    if len(answer) == 0:
                        answer = sample["answerKey"].strip()
                else:
                    reasoning, answer = reasoning.strip(), sample["answerKey"].strip()
                new_data[i][j] = {
                    "query": question,
                    "full_answer": reasoning + "\n####" + answer,
                    "reasoning": reasoning,
                    "answer": answer,
                }
            elif "coin_flip" in dataset_name:
                reasoning = generator.generate_reasoning(
                    sample["inputs"].strip(), dataset_name
                )
                if "####" in reasoning:
                    reasoning, answer = reasoning.split("####")
                    reasoning, answer = reasoning.strip(), answer.strip()
                    if len(answer) == 0:
                        answer = sample["answerKey"].strip()
                else:
                    reasoning, answer = reasoning.strip(), sample["targets"].strip()
                new_data[i][j] = {
                    "query": sample["inputs"].strip(),
                    "full_answer": reasoning + "\n####" + answer,
                    "reasoning": reasoning,
                    "answer": answer,
                }
            if j % 5 == 0 or j == len(data) - 1:
                json.dump(
                    new_data[i],
                    open(path, "w", encoding="utf-8"),
                    indent=2,
                    ensure_ascii=True,
                )
    train_data = json.load(open(train_path, "r", encoding="utf-8"))
    eval_data = json.load(open(eval_path, "r", encoding="utf-8"))
    return train_data, eval_data


def load_datasets(dataset_name, train_data_path, eval_data_path):
    data_path, config_name, val_name = DATASET_MAP[dataset_name]
    generator = ReasoningPairsGenerator()
    train_data, eval_data = preprocess_data(
        dataset_name, data_path, val_name, config_name, train_data_path, eval_data_path
    )
    while len([s for s in train_data if "condensed_reasoning" not in s]) > 0:
        train_data = generator.create_dataset(train_data, train_data_path)
        torch.cuda.empty_cache()
    train_data = RawDataset(train_data, dataset_name)
    eval_data = RawDataset(eval_data, dataset_name)
    return train_data, eval_data


class RawDataset(Dataset):
    """Dataset class for reasoning tasks"""

    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(idx.start, idx.stop)]
        return self.dataset[idx]

    def update_item(self, idx, key, value):
        """Add or update a field in the dataset at the specified index"""
        self.dataset[idx][key] = value


def create_dataloaders(train_dataset, eval_dataset, batch_size=16):
    """Create PyTorch DataLoaders for training and evaluation"""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=3
    )
    return train_loader, eval_loader
