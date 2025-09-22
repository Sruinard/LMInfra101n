from pathlib import Path
import yaml
from pydantic import BaseModel, Field
import logging
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

class SharedModelCfg(BaseModel):
    """Common configuration for model, tokenizer, and artifacts."""
    model_name: str = Field(..., description="Pretrained model identifier or path")
    tokenizer_name: str = Field(..., description="Tokenizer identifier or path")
    artifact_name: str = Field(..., description="Name for the fine-tuned model (for saving/serving)")
    artifact_dir: str = Field(
        "artifacts/", 
        description="The base directory where model artifacts will be written."
    )

class DPOCfg(BaseModel):
    """
    Configuration class for DPO tuning parameters.
    'artifact_dir' is now in SharedModelCfg.
    """
    evaluation_strategy: str = Field("steps")
    eval_steps: int = Field(100)
    save_strategy: str = Field("epoch")
    per_device_train_batch_size: int = Field(1)
    per_device_eval_batch_size: int = Field(1)
    gradient_accumulation_steps: int = Field(16)
    warmup_ratio: float = Field(0.1)
    num_train_epochs: int = Field(2)
    learning_rate: float = Field(5.0e-07)
    logging_steps: int = Field(100)
    optim: str = Field("paged_adamw_8bit")
    lr_scheduler_type: str = Field("cosine")
    seed: int = Field(42)
    max_steps: int = Field(500) # This was 500 in your example
    max_steps: int = Field(-1, description="If > 0: set total number of training steps to perform. Override num_train_epochs.") # Defaulting to -1 like HF
    beta: float = Field(0.05)


class PeftCfg(BaseModel):
    """
    Configuration class for PEFT parameters, based on pydantic.
    (No changes were needed here)
    """
    lora_r: int = Field(16)
    lora_alpha: int = Field(16)
    lora_dropout: float = Field(0.05)
    target_modules: str = Field("all-linear")
    bias: str = Field("none")
    task_type: str = Field("CAUSAL_LM")
    modules_to_save: list = Field(default_factory=lambda: ["lm_head", "embed_tokens"])

class TrainingCfg(BaseModel):
    """
    Configuration class for overall training parameters.
    'model_name' and 'tokenizer_name' are now in SharedModelCfg.
    """
    # REMOVED: model_name
    # REMOVED: tokenizer_name
    dataset_path: str = Field(..., description="Path to the training dataset")
    peft: PeftCfg = Field(..., description="Configuration for PEFT parameters")
    dpo: DPOCfg = Field(..., description="Configuration for DPO tuning parameters")

class ServingCfg(BaseModel):
    """
    Configuration class for serving parameters.
    It can get model info from the parent Config object.
    """
    base_url: str = Field("http://localhost:8000", description="Base URL for the inference server")
    

class Config(BaseModel):
    """
    Main configuration class.
    """
    model: SharedModelCfg = Field(..., description="Shared model and artifact configuration")
    training: TrainingCfg = Field(..., description="TrainingCfg")
    serving: ServingCfg = Field(..., description="ServingCfg")
    

def load_config(env: str, config_path: Path = None) -> Config:
    """
    Load and parse a YAML config file into a Config object based on environment.
    
    Args:
        env: Environment name (e.g. 'dev', 'prod')
        config_path: Optional custom path to config file. If not provided, 
                    defaults to PROJECT_DIR/configs/config.{env}.yaml
        
    Returns:
        Config object with the loaded configuration
    """
    if config_path is None:
        logging.info(f"loading from default config path: {PROJECT_DIR / 'configs' / f'config.{env}.yaml'}")
        config_path = PROJECT_DIR / "configs" / f"config.{env}.yaml"
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
            
    return Config(**config_dict)
