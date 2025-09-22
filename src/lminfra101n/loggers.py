import logging
import sys
from pathlib import Path

def setup_logger(name: str = None, log_file: str = None, level: int = logging.INFO):
    """Set up a global logger with console and optional file handlers.
    
    Args:
        name: Name of the logger. If None, configures the root logger for global use
        log_file: Optional path to log file. If None, only console logging is enabled
        level: Logging level
    
    Returns:
        Logger instance
    """
    # Configure the root logger if no name is provided for global usage
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
