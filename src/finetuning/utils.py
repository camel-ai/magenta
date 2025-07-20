import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--add_easy", action="store_true", default=False)
    parser.add_argument("--error_removed", action="store_true", default=False)
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--repo_name", type=str, default=None)
    return parser.parse_args()

def set_hf_token(token):
    if token:
        os.environ["HF_TOKEN"] = token