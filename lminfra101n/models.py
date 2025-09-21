from transformers import AutoProcessor, AutoModelForCausalLM

def load_model_and_processor(model_name, tokenizer_name):
    tokenizer = AutoProcessor.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    return tokenizer, model