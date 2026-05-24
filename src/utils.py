from datetime import datetime
import logging
import os
from functools import reduce
import yaml


def get_config(key=None, config_path: str = 'config/config.yaml'):
    """Read configuration values from a YAML configuration file.

    Args:
        key: Dot-separated configuration key.
            Example: "index.embedding.model_name".
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary or nested configuration value.

    Raises:
        yaml.YAMLError: If the YAML file cannot be parsed.
        ValueError: If configuration key is not provided.
        KeyError: If the specified key does not exist.
    """
    with open(str(config_path), 'r') as conf:

        try:
            conf = yaml.safe_load(conf)
        except yaml.YAMLError as err:
            print('Error reading configs file: {}'.format(err))

    if key:
        conf = reduce(lambda c, k: c[k], key.split('.'), conf)

    else:
        raise ValueError("Config object not defined")

    return conf


def get_logger(log_dir_path: str) -> logging.Logger:
    """
    Build a logger for a pipeline stage or the orchestrator.

    Creates a new timestamped log file per run and attaches both a
    file handler and a stream handler. If the logger already has
    handlers attached, returns the existing instance.

    Args:
        log_dir_path (str): Directory where the log file is written.
            Created if it does not exist.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Ensure the logging directory exists; if it doesn't, create it
    os.makedirs(log_dir_path, exist_ok=True)

    # Get a logger instance with the given name
    logger = logging.getLogger("logs")

    # Set the logger's severity level to INFO
    logger.setLevel(logging.INFO)

    # Construct the full path for the log file including logger name and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_file = os.path.join(log_dir_path, f"logs_{timestamp}.log")

    # Create a file handler to write logs to the file
    fh = logging.FileHandler(log_file)
    # Set the file handler's severity level to INFO
    fh.setLevel(logging.INFO)

    # Create a stream handler to also output logs to stdout
    sh = logging.StreamHandler()
    # Set the stream handler's severity level to INFO
    sh.setLevel(logging.INFO)

    # Format the file and stream handlers
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Apply the formatter to both file and console handlers
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # Add the file and stream handler to the logger
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Return the fully configured logger instance
    return logger

