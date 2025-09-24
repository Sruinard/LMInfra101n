import argparse
import logging
from lminfra101n import models, configs, loggers, train

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with Gemma model using HuggingFace or OpenAI-compatible API')
    parser.add_argument("--env", type=str, default="dev", help="Environment config to use (dev, prod, etc)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging and load config
    loggers.setup_logger()
    logging.info(f"Starting chat with environment: {args.env}")
    
    cfg = configs.load_config(env=args.env)
    logging.info(f"Loaded configuration for environment: {args.env}")
    
    # Load model and processor
    model_repo = models.RepositoryModel(cfg.model)
    training_service = train.DPOTrainingService(cfg, model_repo)
    logging.info(f"Initialized training service")

    logging.info("Starting training...")
    training_service.run()
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
