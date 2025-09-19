from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

def prepare_dataset(path, tokenizer: AutoTokenizer):
    def _apply_template(example):
        example["prompt"] = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["prompt"]
            }],
            tokenize=False
        )
        example["chosen"] = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "content": example["chosen"]},
            ],
            tokenize=False
        )
        example["rejected"] = tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": example["rejected"]
            }],
            tokenize=False
        )
        return example

    dataset = load_dataset("json", data_files=path)["train"].train_test_split(test_size=0.2)
    dataset["val"] = dataset["test"]
    return dataset.map(_apply_template, num_proc=4)

def supervised_finetuning(dataset, model):
    pass

def dpo_finetuning(dataset, tokenizer, model):
    training_args = TrainingArguments(
            do_eval=True,
            eval_strategy = "steps",
            eval_steps = 1,
            save_strategy = "epoch",
            per_device_train_batch_size = 1, #Zephyr
            gradient_accumulation_steps = 16, #Zephyr
            per_device_eval_batch_size = 1,
            warmup_ratio = 0.1, #Zephyr
            num_train_epochs = 2, #Zephyr
            learning_rate = 5.0e-07, #Zephyr
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 100,
            optim = "paged_adamw_8bit",
            lr_scheduler_type = "cosine", #Zephyr
            seed = 3407,
            output_dir = "./gemma7b_DPO/",
    )


    # from unsloth import PatchDPOTrainer
    # PatchDPOTrainer()
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
    )

    args = DPOConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        max_steps=1,
        logging_steps=10,
        save_steps=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        eval_strategy="steps",
        eval_steps=1,
        output_dir="./results",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        optim="paged_adamw_32bit",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        seed=42,
        beta=0.1
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        peft_config=peft_config,
    )
    return trainer


def merge_for_vllm_serving():
    pass

def tune(cfg):
    pass


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=True, device_map="auto")
    path = "data/dataset.jsonl"
    dataset = prepare_dataset(path, tokenizer)
    trainer = dpo_finetuning(dataset, tokenizer, model)
    trainer.train()
