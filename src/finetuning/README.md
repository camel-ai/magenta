# Finetuning Pipeline for Math LLMs

This directory contains a modular pipeline for supervised finetuning of large language models (LLMs) on mathematical datasets. The design emphasizes clarity, maintainability, and ease of experimentation for math-focused SFT (Supervised Fine-Tuning).

---

## Overview

- **Modular codebase:** Argument parsing, data loading, model/tokenizer setup, and training are separated into distinct modules.
- **LoRA/PEFT support:** Easily configure LoRA rank, dropout, and other PEFT options for efficient finetuning.
- **Custom chat formatting:** Converts math datasets into message-based formats suitable for SFT.
- **Hugging Face Hub integration:** Push checkpoints and final models to your Hugging Face account.
- **Math-centric:** Designed for datasets like MATH and Qwen-generated solutions, but easily adaptable.

---

## Directory Structure

```
src/finetuning/
├── main_finetune.py      # Main orchestration script for finetuning
├── utils.py              # Argument parsing, environment setup
├── data.py               # Dataset loading, merging, formatting
├── train.py              # Training loop, callbacks, memory/reporting
```

---

## Usage

### 1. Install Requirements

```bash
pip install unsloth transformers datasets trl
```

### 2. Prepare Your Hugging Face Token

Set your token as an environment variable or pass it as an argument:
```bash
export HF_TOKEN=your_hf_token
```

### 3. Run Finetuning

From the `src/finetuning/` directory:
```bash
python main_finetune.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_epochs 3 \
  --rank 64 \
  --add_easy \
  --error_removed \
  --cuda_device 0 \
  --max_samples 1000 \
  --repo_name my-math-finetune
```

**Key Arguments:**
- `--model`: Model checkpoint to finetune (default: Qwen/Qwen2.5-7B-Instruct)
- `--train_epochs`: Number of training epochs
- `--rank`: LoRA rank
- `--add_easy`: Include the "easy" dataset in training
- `--error_removed`: Use error-removed solutions
- `--cuda_device`: CUDA device(s) to use
- `--max_samples`: Limit number of samples for quick experiments
- `--repo_name`: Name for output Hugging Face repo

---

## Customization

- **Special Tokens:** Edit `special_tokens_dict` in `tokenization.py` or `main_finetune.py`.
- **Dataset Formatting:** Modify `format_problem_solution_chat` in `data.py`.
- **Training Arguments:** Tune hyperparameters in `train.py` or via CLI.

---

## Outputs

- Model checkpoints and logs are saved in `outputs/{repo_name}/`
- Final model is pushed to the Hugging Face Hub (if configured)

---

## Notes
- The pipeline is modular: you can swap out or extend any stage (data, modeling, training) as needed.
- Designed for math SFT, but can be adapted for other domains with minimal changes.
- For questions or improvements, please open an issue or PR.
