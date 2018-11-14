"""
This module controls the zfit logging.

The base logger for zfit is called `zfit`, and any subsequent logger has the
form `zfit.XX`, where `XX` is its name.

By default, time, name of the logger and message with the default
colorlog color scheme are printed.

"""

import logging
import colorlog


def get_logger(name, lvl=logging.NOTSET, format_=None):
    """Get logger, configuring it on first instantiation.

    If the logger name doesn't start with "zfit", it is automatically added.

    Note:
        Default logging level at first instatiation is INFO.

    Arguments:
        name (str): Name of the logger.
        lvl (int, optional): Logging level. Defaults to `logging.NOTSET`.
        format_ (str): Logger formatting string

    Return:
        `logging.Logger`: The requested logger.

    """
    if not name.startswith('zfit'):
        name = 'zfit.{}'.format(name.rstrip('.'))
    if not format_:
        format_ = ("%(asctime)s - %(name)s | "
                   "%(log_color)s%(levelname)-8s%(reset)s | "
                   "%(log_color)s%(message)s%(reset)s")

    if not logging.root.handlers:
        formatter = colorlog.ColoredFormatter(format_)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logging.root.addHandler(stream)
        logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    return logger
