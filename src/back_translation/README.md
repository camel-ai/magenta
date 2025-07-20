# Back Translation Pipeline

This directory contains a modular pipeline for back-translation and refinement of mathematical solutions, designed for creating high-quality, human-readable datasets from LLM-generated problem solutions.

## Overview

The pipeline consists of two main stages:
1. **Translation and Judgment** (`translate_and_judge.py`):
    - Translates or augments raw solutions, optionally generating tool-call or Chain-of-Thought (CoT) reasoning.
    - Judges or annotates the solutions as needed.
    - Produces an intermediate file with enhanced solutions/logs.
2. **Reformatting** (`reformat.py`):
    - Loads the intermediate results.
    - Reformats and polishes the solutions into a clear, human-style format using prompt-based agents (e.g., GPT-4o-mini, Qwen).
    - Outputs the final, cleaned dataset ready for supervised finetuning or evaluation.

Orchestration is handled by `back_translation_main.py`, which runs both steps in order.

---

## Directory Structure

```
src/back_translation/
├── back_translation_main.py    # Orchestrates the full pipeline
├── translate_and_judge.py      # Stage 1: translation, augmentation, judgment
├── reformat.py                 # Stage 2: reformatting and polishing
├── prompt.py                   # Prompt templates for agents
├── agent_factory.py            # Model/agent creation logic
├── utils.py                    # Utility functions (e.g., saving DataFrames)
```

---

## Usage

### 1. Install Requirements

```bash
pip install openai pandas
# (and any other dependencies required by your agent/model code)
```

### 2. Run the Pipeline

From the `src/back_translation/` directory:

```bash
python back_translation_main.py
```

- This will first run `translate_and_judge.py` and then `reformat.py`.
- Intermediate and final output files will be saved according to the logic in each script (usually as CSV or JSONL).

### 3. Customization

- **Prompts**: Edit `prompt.py` to change system or reformatting prompts.
- **Agent/Model Selection**: Edit or extend `agent_factory.py` to use different LLMs (e.g., OpenAI, Qwen, etc).
- **Input/Output Paths**: Make sure file paths are consistent between stages, or add CLI/config options if needed.

---

## Typical Data Flow

| Stage                   | Script/module           | Input                        | Output                        | Purpose                          |
|-------------------------|------------------------|------------------------------|-------------------------------|-----------------------------------|
| 1. Translation/Judgment | translate_and_judge.py | Raw math problems/solutions  | Enhanced/intermediate results | Tool-call, CoT, or judgment logic |
| 2. Reformatting         | reformat.py            | Enhanced/intermediate results| Final reformatted solutions   | Human-style, polished solutions   |
| Orchestration           | back_translation_main.py| —                            | —                             | Runs both steps in order          |

---

## Notes
- Designed for math SFT dataset creation, but can be adapted for other domains.
- Modular, extensible, and easy to debug.
- Ensure that all dependencies (e.g., OpenAI API keys, model weights) are configured as required by your agent code.

---

For questions, improvements, or issues, please open an issue or PR.
