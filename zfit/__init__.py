"""Top-level package for zfit."""

#  Copyright (c) 2021 zfit
import inspect
import sys
import warnings

from pkg_resources import get_distribution

__version__ = get_distribution(__name__).version

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2018, zfit"
__status__ = "Beta"

__author__ = ("Jonas Eschle <Jonas.Eschle@cern.ch>,"
              "Albert Puig <apuignav@gmail.com>, "
              "Rafael Silva Coutinho <rsilvaco@cern.ch>, "
              "Matthieu Marinangeli <matthieu.marinangeli@cern.ch>")
__maintainer__ = "zfit"
__email__ = 'zfit@physik.uzh.ch'
__credits__ = "Chris Burr, Martina Ferrillo, Abhijit Mathad, Oliver Lantwin, Johannes Lade"

__all__ = ["z", "constraint", "pdf", "minimize", "loss", "core", "data", "func", "dimension", "exception",
           "sample",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]

#  Copyright (c) 2019 zfit

if sys.version_info < (3, 7):
    msg = inspect.cleandoc(
        """zfit is being actively developed and keeps up with the newest versions of other packages.
        This includes Python itself. Therefore, Python 3.6 will be dropped in the near future (beginning of May 2021)
        and 3.9 will be added to the supported versions.

        Feel free to contact us in case of problems to upgrade to a more recent version of Python.
        """
    )
    warnings.warn(msg, FutureWarning, stacklevel=2)


def _maybe_disable_warnings():
    import os
    true = "IS_TRUE"
    if not os.environ.get("ZFIT_DISABLE_TF_WARNINGS", true):
        return
    elif true:
        warnings.warn("All TensorFlow warnings are by default suppressed by zfit."
                      " In order to not suppress them,"
                      " set the environment variable ZFIT_DISABLE_TF_WARNINGS to 0."
                      " In order to suppress the TensorFlow warnings AND this warning,"
                      " set ZFIT_DISABLE_TF_WARNINGS manually to 1.")
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


_maybe_disable_warnings()

import tensorflow as tf

if int(tf.__version__[0]) < 2:
    raise RuntimeError(f"You are using TensorFlow version {tf.__version__}. This zfit version ({__version__}) works"
                       f" only with TF >= 2")

from . import (constraint, core, data, dimension, exception, func, loss,
               minimize, param, pdf, sample, z)
from .core.data import Data
from .core.parameter import (ComplexParameter, ComposedParameter, Parameter,
                             convert_to_parameter)
from .core.space import Space, convert_to_space, supports
from .settings import run, ztypes
from .util.graph import jit as _jit


def _maybe_disable_jit():
    import os
    arg1 = os.environ.get("ZFIT_DO_JIT")
    arg2 = os.environ.get("ZFIT_EXPERIMENTAL_DO_JIT")
    arg3 = os.environ.get("ZFIT_MODE_GRAPH")
    if arg3 is not None:
        warnings.warn("Depreceated to use `ZFIT_MODE_GRAPH`, use `ZFIT_GRAPH_MODE` instead.",
                      DeprecationWarning)

    if arg1 is not None and arg2 is None:
        warnings.warn("Depreceated to use `ZFIT_EXPERIMENTAL_DO_JIT`, use `ZFIT_GRAPH_MODE` instead.",
                      DeprecationWarning)
    arg = arg2 if arg1 is None else arg1
    if arg is not None:
        run.set_graph_mode(bool(int(arg)))

    graph = os.environ.get("ZFIT_GRAPH_MODE")
    if graph is not None:
        run.set_graph_mode(bool(int(graph)))


# experimental flags


_maybe_disable_jit()

# EOF
