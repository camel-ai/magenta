# Solver Agent

## Overview

The `solver_agent` is a crucial component of this project, responsible for both generating Tool-Integrated Reasoning (TIR) traces and evaluating the performance of finetuned models. It is designed to solve mathematical problems from the MATH dataset by leveraging large language models (LLMs) and a suite of specialized toolkits.

At its core, the agent can be configured to use different models, toolkits, and problem sets, making it a flexible tool for a wide range of experiments. The agent's workflow involves loading problems, generating solutions step-by-step, and evaluating the final answers against ground truth solutions.

## Key Features

- **Modular Design**: The agent is composed of several modules, each with a specific responsibility:
    - `main.py`: The main entry point for running the solver agent.
    - `math_loader.py`: Handles loading problems from the MATH dataset.
    - `math_solver.py`: Contains the core logic for solving problems, including interacting with the LLM and toolkits.
    - `evaluator.py`: Evaluates the correctness of the generated solutions.
    - `database.py`: Manages a database for storing experiment results.
    - `prompt.py`: Defines the prompts used to guide the LLM.
    - `schema.py`: Contains the data structures used throughout the agent.
    - `utils.py`: Provides helper functions for various tasks.

- **Toolkit Integration**: The agent supports multiple toolkits to enhance its problem-solving capabilities:
    - **SymPy Toolkit**: For symbolic mathematics.
    - **Code Toolkit**: For executing Python code.
    - **Geometry Toolkit**: For solving geometry problems.

- **Flexible Configuration**: The agent can be configured via command-line arguments to use different models, problem sets, and toolkits, allowing for a wide range of experiments.

## Usage

The solver agent is run from the command line using `main.py`. Below are some examples of how to run the agent.

### Basic Usage

To run the agent on a small set of problems from the "intermediate_algebra" dataset using the `gpt-4o-mini` model, you can use the following command:

```bash
python main.py --model gpt-4o-mini --dataset intermediate_algebra --num 10
```

### Using Toolkits

To enable the SymPy toolkit for solving problems, you can use the `--sympy_toolkit` flag:

```bash
python main.py --model gpt-4o-mini --dataset intermediate_algebra --num 10 --sympy_toolkit
```

Similarly, you can enable the code toolkit with the `--code_toolkit` flag:

```bash
python main.py --model gpt-4o-mini --dataset intermediate_algebra --num 10 --code_toolkit
```

### Command-Line Arguments

Below is a full list of the available command-line arguments for `main.py`:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--level` | str | `None` | Difficulty level of problems to evaluate. |
| `--start_idx` | int | `0` | The starting index of the problems to load. |
| `--num` | int | `10` | The number of problems to evaluate. |
| `--model` | str | `gpt-4o-mini` | The model to use for solving problems. |
| `--checkpoint_path` | str | `None` | Path to a local model checkpoint to use. |
| `--multi_step` | bool | `False` | Use multi-step conversation (no toolkits). |
| `--sympy_toolkit` | bool | `False` | Enable the SymPy toolkit. |
| `--code_toolkit` | bool | `False` | Enable the code toolkit. |
| `--geometry_toolkit`| bool | `False` | Enable the geometry toolkit. |
| `--dataset` | str | `intermediate_algebra` | The MATH dataset to use. |
| `--log` | bool | `False` | Enable logging of results to the database. |
| `--use_csv_problems`| bool | `False` | Use problems from specified CSV files. |
| `--reasoning_path` | str | `...` | Path to the reasoning results CSV file. |
| `--problem_path` | str | `...` | Path to the problem results CSV file. |
| `--mode` | str | `train` | The dataset mode (`train` or `test`). |
| `--port` | int | `8000` | The port for the vLLM server. |
| `--vllm_max_tokens` | int | `8000` | Maximum tokens for the vLLM model output. |
| `--base_path` | str | `...` | Base path for the experiment results CSV files. |
| `--log-file` | str | `None` | Custom path for the log file. |
| `--blacklist_problems`| str | `None` | Path to a CSV file of problems to skip. |
