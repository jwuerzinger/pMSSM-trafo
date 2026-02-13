"""
Logging configuration and utilities for active learning pipelines.

This module provides consistent logging setup across all scripts,
with both file and console output support.
"""

import sys
import logging
from pathlib import Path
import structlog


def setup_logging(timestamp, output_dir=None):
    """
    Set up structlog to write to both file and console.

    Args:
        timestamp: Timestamp string for log file naming
        output_dir: If provided, logs are saved to output_dir/active_learning.log
                   Otherwise, logs are saved to logs/active_learning_{timestamp}.log

    Returns:
        log_file: Path to log file (as string)
        logger: Configured structlog logger
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "active_learning.log"
    else:
        Path("logs/").mkdir(parents=True, exist_ok=True)
        log_file = f"logs/active_learning_{timestamp}.log"

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    return str(log_file), structlog.get_logger()


def setup_worker_logging(log_file_path, model_name):
    """
    Set up logging for a worker process.

    Args:
        log_file_path: Path to the log file
        model_name: Name identifier for this model (e.g., "AL", "Baseline")

    Returns:
        logger: Logger instance for the worker
    """
    # Create a logger specific to this worker
    logger = logging.getLogger(f"worker_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # File handler for this worker
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Format with timestamp and model name
    formatter = logging.Formatter(
        f'%(asctime)s [{model_name}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
