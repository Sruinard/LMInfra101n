import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from lminfra101n.train import prepare_dataset, dpo_finetuning

@pytest.fixture()
def tokenizer():
    return AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml", use_fast=True)

@pytest.fixture()
def model():
    # google/gemma-3-4b-pt
    return AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=True, device_map="auto")

def test_dataset_preparation_splits_and_formats(tokenizer):
    path = "data/dataset.jsonl"
    dataset = prepare_dataset(path, tokenizer)
    assert "train" in dataset
    assert "test" in dataset
    assert "<|im_start|>user\n" in dataset["train"][0]["prompt"]
    assert "<|im_start|>user\n" in dataset["test"][0]["prompt"]

def test_dpo_finetuning_initialization(tokenizer, model):
    path = "data/dataset.jsonl"
    dataset = prepare_dataset(path, tokenizer)
    trainer = dpo_finetuning(dataset, tokenizer, model)
    assert trainer is not None