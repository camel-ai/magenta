from utils import parse_arguments, set_hf_token
from unsloth import FastLanguageModel
from data import load_and_prepare_dataset, prepare_dataset_for_training
from train import train_model
import os


special_tokens_dict = {
    "additional_special_tokens": [
        "<message>", "</message>", "<tool>", "</tool>",
        "<tool_name>", "</tool_name>", "<args>", "</args>", "<cot>", "</cot>"
    ]
}


def load_model_and_tokenizer(args, special_tokens_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    if args.error_removed:
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} special tokens.")
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def force_materialize_model(model, tokenizer):
    import torch
    dummy_input = torch.tensor([[tokenizer.bos_token_id or 0]]).to(model.device)
    with torch.no_grad():
        _ = model(dummy_input)




if __name__ == "__main__":
    args = parse_arguments()
    set_hf_token(args.hf_token)
    model, tokenizer = load_model_and_tokenizer(args, special_tokens_dict)
    force_materialize_model(model, tokenizer)

    SOLUTION = "enhanced_solution_log_error_removed" if args.error_removed else "enhanced_solution_log"
    dataset = load_and_prepare_dataset(args, SOLUTION)
    dataset = prepare_dataset_for_training(dataset, SOLUTION, tokenizer, max_samples=args.max_samples)

    repo_name = args.repo_name or f"{SOLUTION}-{args.add_easy}-{args.rank}-{args.train_epochs}epoch"
    output_dir = os.path.join("outputs", repo_name)
    max_seq_length = 4096
    train_model(model, tokenizer, dataset, args, repo_name, output_dir, max_seq_length)