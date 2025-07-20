# Distilling Tool Knowledge into Language Models via Back-Translated Traces

**This repository contains the code and scripts used for the experiments in the paper:**

[**"Distilling Tool Knowledge into Language Models via Back-Translated Traces"**](https://arxiv.org/abs/2506.19171)  

Please cite the paper if you use this code or data for your research.

---

## Overview

This repository provides a collection of tools and pipelines for synthetic data generation, model finetuning, and evaluation for mathematical reasoning tasks. The codebase is structured into modular components to facilitate experimentation and reproduction of the results presented in the paper.

The codebase is based on CAMEL: [https://github.com/camel-ai/camel](https://github.com/camel-ai/camel).

A comprehensive toolkit for mathematical problem solving and dataset processing, featuring AI-powered math agents and back-translation capabilities for distilling tool knowledge into language models.

## Project Structure

```
math-dataset/
├── src/
│   ├── solver_agent/        # Math problem solving agent
│   ├── back_translation/    # Back translation and reasoning enhancement
│   └── finetuning/         # Model fine-tuning utilities
├── tests/                  # Integration and unit tests
├── scripts/               # Utility scripts for setup and testing
├── data/                  # Dataset files and results
├── logs/                  # Experiment logs and outputs
└── MATH/                  # Math dataset files
```

## Quick Start

### Prerequisites

- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd math-dataset
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Activate the environment**
   ```bash
   source .venv/bin/activate
   ```

## Workflow and Usage

The pipeline follows four main stages as described in the paper:

### 1. Generate Tool-Integrated Reasoning (TIR) Traces

The first step is to use the `solver_agent` to solve problems and generate Tool-Integrated Reasoning (TIR) traces. These traces capture the model's step-by-step reasoning process when using external tools.

```bash
cd src/solver_agent
python main.py --num 10 --dataset algebra --level 1 --model gpt-4o-mini --sympy_toolkit --code_toolkit
```

**Example usage:**
```bash
# Solve 10 algebra problems using GPT-4o-mini with toolkits
python main.py --num 10 --dataset algebra --level 1 --model gpt-4o-mini --sympy_toolkit

# Use with code execution toolkit for computational problems
python main.py --num 5 --dataset intermediate_algebra --code_toolkit --model gpt-4o-mini

# Multi-step reasoning for complex problems
python main.py --num 3 --dataset precalculus --multi_step --model gpt-4o-mini
```

### 2. Back-Translation and Smoothing

Next, the generated TIR traces are processed by the back-translation pipeline located in `src/back_translation/`. This stage refines and polishes the raw traces into high-quality, human-readable solutions suitable for training.

```bash
cd src/back_translation
python back_translation_main.py
```

### 3. Model Finetuning

With the smoothed dataset, you can finetune a language model using the modular scripts in `src/finetuning/`.

```bash
cd src/finetuning

python main_finetune.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --train_epochs 3 \
    --rank 64 \
    --cuda_device 0 \
    --repo_name "my-awesome-math-model" \
    --hf_token "your_hf_token_here"
```

### 4. Evaluation

Finally, the `solver_agent` can be used again to evaluate the finetuned model's performance on standard benchmarks. In this mode, the agent solves problems without the tool-integration and back-translation pipeline to measure its final reasoning capabilities.

### Example: Parallel Finetuning

You can use the same principle to run multiple finetuning jobs with different hyperparameters:

```bash
# Finetune with rank 32 on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/finetuning/main_finetune.py --rank 32 --repo_name model-rank32 &

# Finetune with rank 64 on GPU 1
CUDA_VISIBLE_DEVICES=1 python src/finetuning/main_finetune.py --rank 64 --repo_name model-rank64 &

wait
```

This approach gives you the flexibility to parallelize your workflow on any multi-GPU machine.

## Components

### 1. Solver Agent (`src/solver_agent/`)

The core math-solving component that uses AI models with various toolkits:

- **Multi-step conversation** for complex problem solving
- **SymPy toolkit** for symbolic mathematics
- **Code execution toolkit** for computational problems
- **Geometry toolkit** for geometric problems (when available)
- **Evaluation system** for solution verification

**Key features:**
- Support for multiple AI models (OpenAI, Qwen, etc.)
- Comprehensive logging and database storage
- Configurable toolkits and solving strategies
- Real-time performance metrics

### 2. Back Translation (`src/back_translation/`)

Enhances mathematical reasoning by generating explanations and verifying solutions:

- **Solution enhancement** with detailed explanations
- **Reasoning quality assessment** 
- **Chain-of-thought generation**
- **Solution verification** using multiple models

### 3. Fine-tuning (`src/finetuning/`)

Tools for training and fine-tuning models on mathematical datasets:

- **Dataset preparation** and preprocessing
- **Training pipelines** for various model architectures
- **Evaluation metrics** and benchmarking
- **Model optimization** techniques

## Testing

Run the comprehensive integration test suite:

```bash
./scripts/run_tests.sh
```

This will test:
- Math agent initialization and problem solving
- Back translation workflow
- Component integration
- End-to-end functionality (with API key)

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=your_openai_api_key

# Optional: Other API keys for different models
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
SAMBA_API_KEY=your_samba_key
```

### Model Configuration

The project supports various AI models:

- **OpenAI models**: `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`
- **Qwen models**: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-Math-7B`
- **Other models**: Mistral, Groq, etc.

### Toolkit Configuration

Available toolkits for enhanced problem solving:

- **SymPy**: Symbolic mathematics and equation solving
- **Code Execution**: Python code execution for computational problems
- **Math**: Basic arithmetic operations
- **Geometry**: Geometric problem solving (when available)

## Dataset Support

The project works with various mathematical datasets:

- **MATH**: Competition mathematics dataset
- **GSM8K**: Grade school math word problems
- **AIME**: American Invitational Mathematics Examination
- **AMC**: American Mathematics Competitions
- **Custom datasets**: Support for custom problem formats

## Development

### Project Structure

- `src/solver_agent/`: Core math-solving logic
- `src/back_translation/`: Reasoning enhancement tools
- `src/finetuning/`: Model training utilities
- `tests/`: Integration and unit tests
- `scripts/`: Setup and utility scripts

### Adding New Features

1. **New Toolkits**: Add to `src/solver_agent/math_solver.py`
2. **New Models**: Configure in model initialization
3. **New Datasets**: Extend `src/solver_agent/math_loader.py`
4. **Tests**: Add integration tests in `tests/`

### Code Style

The project follows Python best practices:
- Type hints where applicable
- Comprehensive logging
- Error handling and validation
- Modular design for extensibility

## Performance and Metrics

The system tracks various performance metrics:

- **Accuracy**: Percentage of correctly solved problems
- **Tool Usage**: Which toolkits were employed
- **Solving Time**: Time taken per problem
- **Error Analysis**: Types and frequencies of errors

Results are stored in:
- Database (SQLite) for structured data
- CSV files for analysis
- Log files for debugging

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **API Errors**: Check API keys in `.env` file
3. **Missing Data**: Download required datasets to `MATH/` directory
4. **Permission Errors**: Make scripts executable with `chmod +x`

### Getting Help

- Check the logs in `logs/` directory
- Run tests to verify setup: `./scripts/run_tests.sh`
- Review component-specific READMEs in `src/*/README.md`

## Citation

If you find this work useful, please cite the following paper:

```bibtex
@inproceedings{huang2025distillingtoolknowledgelanguage,
      title={Distilling Tool Knowledge into Language Models via Back-Translated Traces}, 
      author={Xingyue Huang and Xianglong Hu and Zifeng Ding and Yuan He and Rishabh and Waleed Alzarooni and Ziyu Ye and Wendong Fan and Bailan He and Haige Bo and Changran Hu and Guohao Li},
      year={2025},
      booktitle={ICML 2025 Workshop on Multi-Agent Systems in the Era of Foundation Models: Opportunities, Challenges and Futures}, 
}
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]