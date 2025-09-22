import argparse
import logging
from lminfra101n import models, configs, loggers, train
import torch
import requests

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with Gemma model using HuggingFace or OpenAI-compatible API')
    parser.add_argument("--env", type=str, default="dev", help="Environment config to use (dev, prod, etc)")
    parser.add_argument("--mode", type=str, default="hf", choices=['hf', 'openai'], 
                       help="Mode to run in - 'hf' for HuggingFace or 'openai' for OpenAI-compatible API")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging and load config
    loggers.setup_logger()
    logging.info(f"Starting chat with environment: {args.env}")
    
    cfg = configs.load_config(env=args.env)
    logging.info(f"Loaded config: {cfg.model_name} and {cfg.processor_name}")
    
    # Load model and processor
    processor, model = models.load_model_and_processor(cfg.model_name, cfg.processor_name)
    logging.info(f"Loaded model and processor")


    logging.info(f"Preparing dataset...")
    dataset = train.prepare_dataset(cfg.dataset_path, processor)
    logging.info("done.")


    logging.info(f"Training model...")
    trainer = train.dpo_finetuning(dataset, processor, model)
    logging.info("done.")

    logging.info(f"Training model...")
    trainer.train()

    trainer.save_model("./results/checkpoint-custom")
    logging.info("done.")


if __name__ == "__main__":
    main()
