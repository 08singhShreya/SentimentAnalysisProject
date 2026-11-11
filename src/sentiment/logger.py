"""Logging setup shared by the whole project.

Every module does:  logger = get_logger(__name__)
so all messages share one format and one level.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to the terminal.

    Safe to call many times with the same name. Without the handler
    check, each call would add another handler and every line would
    print twice.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  #Keeps the message from being printed a second time by Python’s default root logger.
    return logger