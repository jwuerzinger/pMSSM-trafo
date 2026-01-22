"""
Test script to verify structlog logging works correctly.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

from pathlib import Path
import sys
from datetime import datetime
import logging
import structlog


def setup_logging():
    """Set up structlog to write to both file and console."""
    Path("logs/").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/test_{timestamp}.log"

    # Configure standard library logging to write to file and console
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure structlog for human-readable output
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up formatters for console and file
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    # Apply formatter to handlers
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    return log_file, structlog.get_logger()


if __name__ == "__main__":
    log_file, logger = setup_logging()

    logger.info("="*60)
    logger.info("Testing structlog configuration")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)

    logger.info("This message should appear in both console and log file")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("\nTest complete!")
    logger.info(f"Check {log_file} to verify logging works")

    print(f"\n✓ Log file created at: {log_file}")
