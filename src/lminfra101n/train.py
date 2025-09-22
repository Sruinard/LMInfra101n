from lminfra101n import configs, models
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig


def prepare_dataset(path, tokenizer: AutoTokenizer):
    def _apply_template(example):
        # tokenizer chat_template expects user/assistant/user/assistant
        prompt, chosen = tokenizer.apply_chat_template([
            { "role": "system", "content": "you are an ML Academy AI." },
            { "role": "user", "content": example["prompt"] },
            { "role": "assistant", "content": example["chosen"] },
        ], tokenize=False).split("<start_of_turn>model\n")
        prompt, rejected = tokenizer.apply_chat_template([
            { "role": "system", "content": "you are an ML Academy AI." },
            { "role": "user", "content": example["prompt"] },
            { "role": "assistant", "content": example["rejected"]},
        ], tokenize=False).split("<start_of_turn>model\n")

        example["prompt"] = prompt.replace("<bos>", "")
        # split removed <start_of_turn>model\n from the chosen and rejected
        example["chosen"] = f"<start_of_turn>model\n{chosen}"
        example["rejected"] = f"<start_of_turn>model\n{rejected}"
        return example
 

    dataset = load_dataset("json", data_files=path)["train"].train_test_split(test_size=0.2)
    # rename test -> val and remove test
    dataset["val"] = dataset["test"]
    dataset.pop("test")
    return dataset.map(_apply_template, num_proc=4)


def _create_lora_config(cfg: configs.PeftCfg) -> LoraConfig:
    return LoraConfig(
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        r=cfg.lora_r,
        bias=cfg.bias,
        target_modules=cfg.target_modules,
        task_type=cfg.task_type,
        modules_to_save=cfg.modules_to_save
    )
def _create_dpo_training_args(cfg: configs.DPOCfg) -> DPOConfig:
    return DPOConfig(
        eval_strategy=cfg.evaluation_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        beta=cfg.beta,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    )

class DPOTrainingService:
    def __init__(self, cfg: configs.Config, model_repo: models.RepositoryModel):
        self.cfg = cfg
        self.model_repo = model_repo

    def _init_dpo_trainer(self):
        peft_config = _create_lora_config(self.cfg.training.peft)
        dpo_args = _create_dpo_training_args(self.cfg.training.dpo)
        dpo_args.output_dir = self.cfg.model.artifact_dir

        tokenizer, model = self.model_repo.load_tokenizer_and_model()
        dataset = prepare_dataset(self.cfg.training.dataset_path, tokenizer)

        trainer = DPOTrainer(
            model,
            ref_model=None,
            args=dpo_args,
            processing_class=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            peft_config=peft_config,
        )
        return trainer
    
    def run(self):
        trainer = self._init_dpo_trainer()
        trainer.train()
        self.model_repo.save_tokenizer_and_peft_model()
        