"""This module controls the zfit logging.

The base logger for zfit is called `zfit`, and all loggers created by this module
have the form `zfit.XX`, where `XX` is their name.

By default, time, name of the logger and message with the default
colorlog color scheme are printed.
"""

#  Copyright (c) 2022 zfit

import logging
import os

import colorlog


def get_logger(name, stdout_level=None, file_level=None, file_name=None):
    """Get and configure logger.

    This logger has two handlers:
        - A stdout handler is always configured with `colorlog`.
        - A file handler is configured if `file_name` is given. Once it has been configure, it is not
        necessary to give it to modify its properties.

    Once the logger has been created, `get_logger` can be called again to modify its log levels,
    independently for the stream and file handlers.

    Note:
        If the logger name doesn't start with "zfit", it is automatically added.

    Note:
        Default logging level at first instantiation is WARNING.

    Arguments:
        name: Name of the logger.
        stdout_level: Logging level for the stream handler. Defaults to `logging.WARNING`.
        file_level: Logging level for the file handler. Defaults to `logging.WARNING`.
        file_name: File to log to. If not given, no file logging is performed.

    Return:
        `logging.Logger`: The requested logger.

    Raise:
        ValueError if `file_level` has been specified without having configured the output file.
    """
    if not name.startswith("zfit"):
        name = "zfit.{}".format(name.rstrip("."))
    if stdout_level is None:
        stdout_level = logging.WARNING
    format_stream = (
        "%(asctime)s - %(name)s | "
        "%(log_color)s%(levelname)-8s%(reset)s | "
        "%(log_color)s%(message)s%(reset)s"
    )
    format_file = "%(asctime)s - %(name)s | " "%(levelname)-8s | " "%(message)s"
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Add Stream handler
        formatter = colorlog.ColoredFormatter(format_stream)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    # The first handler is always the stream

    logger.handlers[0].setLevel(stdout_level)
    # Now the file handler
    file_handler = None
    # Find the file handler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if not file_name or os.path.abspath(file_name) == handler.baseFilename:
                file_handler = handler
                break
    # If requested, create one
    if file_name and file_handler is None:
        formatter = colorlog.ColoredFormatter(format_file)
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Set its level
    if file_level is not None and file_handler is None:
        raise ValueError(
            "Requested change in file log level but no file logger has been  configured"
        )
    if file_level is None:
        file_level = logging.WARNING
    if file_handler is not None:
        file_handler.setLevel(file_level)
    # Set the logging level to the lowest level
    logger_level = min(stdout_level, file_level)
    logger.setLevel(logger_level)
    return logger
