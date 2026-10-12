"""
src/utils/logging_utils.py
===========================
Shared logging configuration for all project scripts.

Handles Windows cp1252 encoding when logging Telugu Unicode text.
"""

import sys
import logging
import pathlib
import datetime


def configure_utf8_console():
    """Reconfigure stdout/stderr to UTF-8 on Windows to handle Telugu text."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str, log_file: pathlib.Path = None) -> logging.Logger:
    """
    Return a logger with console + optional file handler.
    Always calls configure_utf8_console() to prevent cp1252 errors.
    """
    configure_utf8_console()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler (UTF-8 reconfigured)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (always UTF-8)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
