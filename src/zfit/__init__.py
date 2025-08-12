"""Top-level package for zfit."""
#  Copyright (c) 2025 zfit
from __future__ import annotations

import contextlib as _contextlib
import logging as _logging
import os as _os
import sys as _sys
import typing
from importlib.metadata import version as _importlib_version

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__version__ = _importlib_version(__name__)

__license__ = "BSD 3-Clause"
__copyright__ = "2024, zfit"
__status__ = "Beta"

__author__ = (
    "Jonas Eschle <Jonas.Eschle@cern.ch>,"
    "Albert Puig <albert@protonmail.com>, "
    "Rafael Silva Coutinho <rsilvaco@cern.ch>, "
    "Matthieu Marinangeli <matthieu.marinangeli@cern.ch>"
)
__maintainer__ = "zfit"
__email__ = "zfit@physik.uzh.ch"
__credits__ = "Chris Burr, Martina Ferrillo, Abhijit Mathad, Oliver Lantwin, Johannes Lade, Iason Krommydas"

__all__ = [
    "z",
    "constraint",
    "pdf",
    "minimize",
    "loss",
    "dill",
    "data",
    "Data",
    "func",
    "binned",
    "dimension",
    "exception",
    "interface",
    "sample",
    "binned",
    "hs3",
    'param',
    "Parameter",
    "ComposedParameter",
    "ComplexParameter",
    "convert_to_parameter",
    "Space",
    "convert_to_space",
    "supports",
    "result",
    "run",
    "settings",
    "ztypes",
    "prior",
    "mcmc",
    "Data",
    "ztypes",
    "param",
]


#  Copyright (c) 2019 zfit


@_contextlib.contextmanager
def _suppress_stderr():
    original_stderr_fd = _sys.stderr.fileno()
    # Duplicate the original stderr file descriptor
    saved_stderr_fd = _os.dup(original_stderr_fd)
    try:
        # Open /dev/null and redirect stderr to it
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(devnull_fd, original_stderr_fd)
        _os.close(devnull_fd)
        yield
    finally:
        # Restore the original stderr
        _os.dup2(saved_stderr_fd, original_stderr_fd)
        _os.close(saved_stderr_fd)


def _maybe_disable_warnings() -> None:
    import os
    import warnings

    disable_warnings = os.environ.get("ZFIT_DISABLE_TF_WARNINGS")
    if disable_warnings is None:
        warnings.warn(
            "TensorFlow warnings are by default suppressed by zfit."
            " In order to show them,"
            " set the environment variable ZFIT_DISABLE_TF_WARNINGS=0."
            " In order to suppress the TensorFlow warnings AND this warning,"
            " set ZFIT_DISABLE_TF_WARNINGS=1."
        )
    elif disable_warnings == "0":  # just ignore and do nothing
        return

    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        import tensorflow as tf

        tf.constant(1)
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        tf.get_logger().setLevel("ERROR")
    # with redirect_stdout(None), redirect_stderr(None):
    #
    #     logging.getLogger("tensorflow").disabled = True
    #     import tensorflow as tf
    #
    #     tf.constant(1)


    # os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    _logging.getLogger("absl").setLevel(_logging.ERROR)
    _logging.getLogger("tensorflow").setLevel(_logging.FATAL)
    # raise RuntimeError
    with _suppress_stderr():
        import tensorflow as tf
    #
    tf.get_logger().setLevel("FATAL")
    tf.autograph.set_verbosity(0)


_maybe_disable_warnings()

import tensorflow as _tf

if int(_tf.__version__[0]) < 2:
    raise RuntimeError(
        f"You are using TensorFlow version {_tf.__version__}. This zfit version ({__version__}) works only with TF >= 2"
    )
from . import (
    binned,
    constraint,
    data,
    dill,
    dimension,
interface,
    exception,
    func,
    hs3,
    loss,
    minimize,
    mcmc,
    param,
    pdf,
    prior,
    result,
    sample,
    settings,
    z,  # initialize first
)
from .core.data import Data
from .core.parameter import (
    ComplexParameter,
    ComposedParameter,
    Parameter,
    convert_to_parameter,
)
from .core.space import Space, convert_to_space, supports
from .settings import run, ztypes


def _maybe_disable_jit() -> None:
    import os
    import warnings

    arg1 = os.environ.get("ZFIT_DO_JIT")
    arg2 = os.environ.get("ZFIT_EXPERIMENTAL_DO_JIT")
    arg3 = os.environ.get("ZFIT_MODE_GRAPH")
    if arg3 is not None:
        warnings.warn(
            "Depreceated to use `ZFIT_MODE_GRAPH`, use `ZFIT_GRAPH_MODE` instead.",
            DeprecationWarning,
        )

    if arg1 is not None and arg2 is None:
        warnings.warn(
            "Depreceated to use `ZFIT_EXPERIMENTAL_DO_JIT`, use `ZFIT_GRAPH_MODE` instead.",
            DeprecationWarning,
        )
    arg = arg2 if arg1 is None else arg1
    if arg is not None and not int(arg):
        run.set_graph_mode(False)

    graph = os.environ.get("ZFIT_GRAPH_MODE")
    if graph is not None and not int(graph):
        run.set_graph_mode(False)


# experimental flags

_maybe_disable_jit()

# EOF
