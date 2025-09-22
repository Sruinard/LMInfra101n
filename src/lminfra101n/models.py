import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

def load_model_and_processor(model_name, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    # MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[model.config.model_type] = "Gemma3ForConditionalGeneration"
    return tokenizer, model

def load_peft_model(model_name):
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    return model

def merge_peft_and_base_model(model_id, output_dir):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    # Load Model base model
    model = AutoModelForCausalLM.from_pretrained(model_id,  trust_remote_code=True, low_cpu_mem_usage=True)

    # Merge LoRA and base model and save
    peft_model = PeftModel.from_pretrained(model, output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

    processor = AutoTokenizer.from_pretrained(output_dir)
    processor.save_pretrained("merged_model")

if __name__ == "__main__":
    # model = load_peft_model("results/checkpoint-1")
    merge_peft_and_base_model("google/gemma-3-1b-it", "results/checkpoint-1")