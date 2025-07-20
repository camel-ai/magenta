from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import torch

class PushToHubEachEpochCallback(TrainerCallback):
    def __init__(self, repo_name):
        self.repo_name = repo_name
    def on_epoch_end(self, args, state, control, **kwargs):
        kwargs["model"].push_to_hub(
            self.repo_name,
            commit_message=f"Checkpoint after epoch {int(state.epoch)}"
        )

def train_model(model, tokenizer, dataset, args, repo_name, output_dir, max_seq_length):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=getattr(args, "batch_size", 2),
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.train_epochs,
            max_steps=1,
            learning_rate=2e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            save_strategy="epoch",
            push_to_hub=True,
            hub_model_id=repo_name,
        ),
    )
    trainer.add_callback(PushToHubEachEpochCallback(repo_name))

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.push_to_hub_merged(repo_name, tokenizer, save_method="merged_16bit")