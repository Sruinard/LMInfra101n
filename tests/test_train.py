import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from lminfra101n import train, configs, models

@pytest.fixture()
def config() -> configs.Config:
    return configs.load_config("dev")

@pytest.fixture()
def processor_and_model(config: configs.Config):
    return models.load_model_and_processor(config.model_name, config.processor_name)

def test_gemma_3_1b_it_loading():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    assert tokenizer is not None

def test_consistent_token_assignment(processor_and_model: tuple[AutoTokenizer, AutoModelForCausalLM]):
    tokenizer, _ = processor_and_model
    features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
    tokenizer(features["prompt"], add_special_tokens=False)


def test_dataset_preparation_splits_and_formats(config: configs.Config, processor_and_model: tuple[AutoTokenizer, AutoModelForCausalLM]):
    tokenizer, _ = processor_and_model
    path = config.dataset_path
    dataset = train.prepare_dataset(path, tokenizer)
    assert "train" in dataset
    assert "test" in dataset
    assert "<|im_start|>user\n" in dataset["train"][0]["prompt"]
    assert "<|im_start|>user\n" in dataset["test"][0]["prompt"]

def test_dpo_finetuning_initialization(tokenizer, model):
    path = "data/dataset.jsonl"
    dataset = prepare_dataset(path, tokenizer)
    trainer = dpo_finetuning(dataset, tokenizer, model)
    assert trainer is not None