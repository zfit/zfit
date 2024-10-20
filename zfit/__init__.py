"""Top-level package for zfit."""

#  Copyright (c) 2024 zfit

from importlib.metadata import version as _importlib_version

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
__credits__ = (
    "Chris Burr, Martina Ferrillo, Abhijit Mathad, Oliver Lantwin, Johannes Lade, Iason Krommydas"
)

__all__ = [
    "z",
    "constraint",
    "pdf",
    "minimize",
    "loss",
    "dill",
    "data",
    "func",
    "dimension",
    "exception",
    "sample",
    "binned",
    "hs3",
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
]


#  Copyright (c) 2019 zfit


def _maybe_disable_warnings():
    import os, warnings

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

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")


_maybe_disable_warnings()

import tensorflow as _tf

if int(_tf.__version__[0]) < 2:
    raise RuntimeError(
        f"You are using TensorFlow version {_tf.__version__}. This zfit version ({__version__}) works"
        f" only with TF >= 2"
    )
from . import z  # initialize first
from . import (
    constraint,
    data,
    dimension,
    exception,
    func,
    dill,
    loss,
    binned,
    minimize,
    param,
    pdf,
    result,
    sample,
    settings,
    hs3,
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


def _maybe_disable_jit():
    import os, warnings

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
