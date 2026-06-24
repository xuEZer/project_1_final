"""Logging utilities for console and file outputs."""

import logging
import sys
from configparser import ConfigParser
from pathlib import Path

import torch


class ColoredFormatter(logging.Formatter):
    """Formatter that applies ANSI colors based on log level."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"

    FORMATS = {
        logging.DEBUG: (
            grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
        ),
        logging.INFO: (
            blue
            + "%(asctime)s"
            + reset
            + " - "
            + green
            + "%(levelname)s"
            + reset
            + " - %(message)s"
        ),
        logging.WARNING: (yellow + "%(asctime)s - %(levelname)s - %(message)s" + reset),
        logging.ERROR: red + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: (
            bold_red + "%(asctime)s - %(levelname)s - %(message)s" + reset
        ),
    }

    def format(self, record):
        """Format a log record with a level-specific colored format string."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    log_file_path: Path,
    log_name: str,
    mode: str = "w",
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """Create and configure a logger with optional file and console handlers.

    Args:
        log_file_path: Log file output path.
        log_name: Logger name.
        mode: File open mode for the file handler.
        level: Logging level string.
        enable_console: Whether to enable console logging.
        enable_file: Whether to enable file logging.

    Returns:
        Configured logger instance.

    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger = logging.getLogger(log_name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    if enable_file:
        fh = logging.FileHandler(log_file_path, mode=mode)
        fh.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    if enable_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

    logger.propagate = False

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log runtime system information relevant for reproducibility.

    Args:
        logger: Logger used to emit system information.

    Returns:
        None.

    """
    logger.info("=" * 30 + " System Info " + "=" * 30)
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 75)


def log_config(config: ConfigParser, logger: logging.Logger) -> None:
    """Log all parsed configuration sections and key-value pairs.

    Args:
        config: Parsed configuration object.
        logger: Logger used to emit configuration values.

    Returns:
        None.

    """
    logger.info("=" * 30 + " Configuration " + "=" * 30)
    for section in config.sections():
        logger.info(f"[{section}]")
        for key, value in config.items(section):
            logger.info(f"  {key} = {value}")
    logger.info("=" * 75)
