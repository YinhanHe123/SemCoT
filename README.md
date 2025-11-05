# SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens

Official PyTorch implementation of the NeurIPS 2025 paper: "SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens" [View Paper (PDF)](https://www.arxiv.org/pdf/2510.24940)

<img width="840" height="491" alt="Screenshot 2025-10-25 at 9 21 17 AM" src="https://github.com/user-attachments/assets/e7209cc2-e0c9-48a1-964a-933ead85d9f8" />

## Overview

SemCoT is a novel framework that accelerates Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) by replacing verbose explicit reasoning with compact, semantically-aligned implicit tokens. Unlike existing methods, SemCoT jointly optimizes both token-level generation speed and semantic alignment with ground-truth reasoning.

### Key Features

- **Semantic Alignment**: Custom sentence transformer ensures implicit reasoning preserves the semantics of ground-truth reasoning
- **Efficient Generation**: Lightweight language model generates implicit tokens faster than full LLMs
- **Superior Performance**: Achieves state-of-the-art results across multiple reasoning benchmarks
- **Flexible Architecture**: Works with various LLM backbones (Llama-2, Mistral, Qwen)

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YinhanHe123/SemCoT.git
cd SemCoT

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
transformers==4.57.0
peft==0.17.1
datasets==4.0.0
tensorboardX==2.6.2.2
```

## Quick Start

### 1. Generate Dataset with GPT-4o-mini

First, generate the reasoning pairs dataset:

```bash
python main.py \
  --mode generate_data \
  --dataset gsm8k \
  --config small \
  --device 0
```

**Note**: You'll need to add your OpenAI API key in `data/gpt4pair.py`:
```python
def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = "your-api-key-here"):
```

### 2. Train and Evaluate SemCoT

```bash
python main.py \
  --mode semcot \
  --dataset gsm8k \
  --config small \
  --use_best_params \
  --num_exps 3 \
  --device 0
```

### 3. Run Baseline Methods

```bash
python main.py \
  --mode baseline \
  --baseline pause \
  --dataset gsm8k \
  --config small \
  --num_exps 3 \
  --device 0
```

## Datasets

SemCoT supports five reasoning datasets:

- **Mathematical Reasoning**: GSM8K, SVAMP, MultiArith
- **Commonsense Reasoning**: CommonsenseQA
- **Symbolic Reasoning**: CoinFlip

Datasets are automatically downloaded and preprocessed on first use.

## Model Configurations

Three LLM configurations are available:

| Config | Teacher Model | Student Model | Use Case |
|--------|--------------|---------------|----------|
| `small` | Llama-2-7B | Sheared-LLaMA-1.3B | Default |
| `mistral` | Mistral-7B | mistral-1.1b | Alternative |
| `qwen` | Qwen2.5-7B | Qwen2.5-0.5B | Experimental |

## Command Line Arguments

### Core Arguments

- `--mode`: Operation mode (`semcot`, `baseline`, `generate_data`)
- `--config`: Model configuration (`small`, `mistral`, `qwen`)
- `--dataset`: Dataset name (`gsm8k`, `svamp`, `multiarith`, `commonsense_qa`, `coin_flip`)
- `--baseline`: Baseline method (`pause`, `icot_si`, `codi`, `softcot`, `coconut`)
- `--device`: GPU device ID
- `--num_exps`: Number of experimental runs (default: 3)
- `--use_best_params`: Use pre-tuned hyperparameters

### Training Hyperparameters

**Sentence Transformer:**
- `--st_linear_lr`: Linear layer learning rate (default: 1e-4)
- `--st_linear_wd`: Linear layer weight decay (default: 1e-3)
- `--st_linear_epochs`: Linear layer epochs (default: 7)
- `--st_llm_lr`: LLM learning rate (default: 1e-7)
- `--st_llm_wd`: LLM weight decay (default: 1e-5)
- `--st_llm_epochs`: LLM epochs (default: 3)

**Contemplation Generator:**
- `--cg_linear_lr`: Linear layer learning rate (default: 1e-4)
- `--cg_linear_wd`: Linear layer weight decay (default: 1e-3)
- `--cg_linear_epochs`: Linear layer epochs (default: 7)
- `--cg_llm_lr`: LLM learning rate (default: 1e-7)
- `--cg_llm_wd`: LLM weight decay (default: 1e-5)
- `--cg_llm_epochs`: LLM epochs (default: 3)
- `--alpha`: Balance between semantic alignment and answer accuracy (default: 0)

### Other Parameters

- `--train_max_contemp_tokens`: Number of implicit tokens during training (default: 5)
- `--eval_max_contemp_tokens`: Number of implicit tokens during evaluation (default: 1)
- `--batch_size`: Training batch size (default: 4)
- `--seed`: Random seed (default: 42)

## Ablation Studies

SemCoT includes several ablation variants:

```bash
# Without sentence transformer
python main.py --mode semcot --variation no_sentence_transformer

# Without semantic alignment loss
python main.py --mode semcot --variation no_l_reason

# Without warmup training
python main.py --mode semcot --variation no_warmup

# Using full LLM instead of lightweight model
python main.py --mode semcot --variation no_small_contemp_gen
```

## Results

Results are saved in `results/{mode}/{variation}/{config}/{dataset}/`:
- **Logs**: Training metrics and console output
- **Results**: JSONL files with accuracy and timing metrics
- **Models**: Trained model checkpoints

### Example Output

```json
{
  "numerical_accuracy": 0.983,
  "ave_sample_time": 1.02,
  "dataset": "gsm8k",
  "eval_temp": 0.7,
  "exp_num": 0
}
```

## Project Structure

```
SemCoT/
├── data/                   # Dataset handling
│   ├── cot_datasets.py    # Dataset loaders
│   └── gpt4pair.py        # GPT-4 reasoning generation
├── models/                 # Model implementations
│   ├── semcot.py          # SemCoT components
│   ├── pause.py           # Baseline: Pause tokens
│   ├── icot_si.py         # Baseline: ICoT-SI
│   ├── codi.py            # Baseline: CODI
│   ├── softcot.py         # Baseline: SoftCoT
│   └── coconut.py         # Baseline: COCONUT
├── training/               # Training scripts
│   ├── train_semcot.py    # SemCoT training
│   └── train_*.py         # Baseline training
├── utils/                  # Utilities
│   ├── logging.py         # TensorBoard logging
│   └── utils.py           # Helper functions
├── main.py                 # Main entry point
└── requirements.txt        # Dependencies
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{he2025semcot,
  title={SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens},
  author={He, Yinhan and Zheng, Wendy and Zhu, Yaochen and Zheng, Zaiyi and Su, Lin and Vasudevan, Sriram and Guo, Qi and Hong, Liangjie and Li, Jundong},
  booktitle={39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### LLM Licenses

The models used in this project have their own licenses:
- **Llama-2**: [Llama 2 Community License](LLM%20Licenses/Llama-7b-chat-hf%20license.rtf)
- **Mistral**: [Apache 2.0](LLM%20Licenses/Mistral-7B-Instruct-v0.2%20license)
- **Sheared-LLaMA**: [Apache 2.0](LLM%20Licenses/sheared-llama-1.3b%20license.rtf)

Please ensure compliance with the respective model licenses when using this code.

## Acknowledgments

This work was supported by the National Science Foundation (NSF), Office of Naval Research (ONR), and Commonwealth Cyber Initiative (CCI). We thank the authors of the baseline methods for their open-source implementations.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Yinhan He: nee7ne@virginia.edu
- Jundong Li: jl6qk@virginia.edu
