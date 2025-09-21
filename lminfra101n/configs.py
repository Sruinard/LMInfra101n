from pathlib import Path
import yaml
from pydantic import BaseModel, Field
import logging
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent

class Config(BaseModel):
    model_name: str = Field(..., description="Name/path of the model to load")
    processor_name: str = Field(..., description="Name/path of the processor to load")
    dataset_path: str = Field(..., description="Path to the dataset")

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
