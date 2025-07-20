from datasets import load_dataset, concatenate_datasets, Value

def load_and_prepare_dataset(
    args, 
    solution_col, 
    DATASET_NAME = "hxyscott/math-dataset-decontamination-4.1-mini",
    EASY_DATASET_NAME = "hxyscott/math-dataset-decontamination-4.1-mini-easy",
):
    SPLIT = "train"
    print(f"Loading dataset '{DATASET_NAME}' split '{SPLIT}'...")
    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT)
        required_columns = ["problem", solution_col]
        if not all(col in dataset.column_names for col in required_columns):
            raise ValueError(f"Dataset missing one or more required columns: {required_columns}. Found: {dataset.column_names}")
        print(f"Dataset loaded successfully. Columns: {dataset.column_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Optionally add easy dataset
    if getattr(args, "add_easy", False):
        print(f"Loading easy dataset '{EASY_DATASET_NAME}' split '{SPLIT}'...")
        try:
            easy_dataset = load_dataset(EASY_DATASET_NAME, split=SPLIT)
            required_columns = ["problem", solution_col]
            if not all(col in easy_dataset.column_names for col in required_columns):
                raise ValueError(f"Easy dataset missing required columns: {required_columns}. Found: {easy_dataset.column_names}")
            print(f"Easy dataset loaded successfully. Columns: {easy_dataset.column_names}")
        except Exception as e:
            print(f"Error loading easy dataset: {e}")
            raise
        # Ensure 'level' column is string in both datasets before concatenation
        if "level" in dataset.column_names:
            dataset = dataset.cast_column("level", Value("string"))
        if "level" in easy_dataset.column_names:
            easy_dataset = easy_dataset.cast_column("level", Value("string"))
        dataset = concatenate_datasets([dataset, easy_dataset])
        print(f"Merged dataset with easy dataset. Total rows: {len(dataset)}")
        dataset = dataset.shuffle(seed=3407)
    return dataset

def format_problem_solution_chat(example, solution_col, tokenizer):
    problem_text = example["problem"]
    solution_text = example[solution_col]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solve math problems. Please wrap your final answer in \\boxed{}."},
        {"role": "user", "content": problem_text},
        {"role": "assistant", "content": solution_text}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_text}

def prepare_dataset_for_training(dataset, solution_col, tokenizer, max_samples=None):
    columns_to_remove = list(dataset.column_names)
    dataset = dataset.map(
        lambda ex: format_problem_solution_chat(ex, solution_col, tokenizer),
        remove_columns=columns_to_remove
    )
    if max_samples is not None:
        print(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print("Dataset formatting complete.")
    if len(dataset) > 0:
        print("Example formatted text:\n", dataset[0]['text'])
    else:
        print("Formatted dataset is empty.")
    return dataset