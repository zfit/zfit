"""Top-level package for zfit."""

#  Copyright (c) 2021 zfit
import warnings

from pkg_resources import get_distribution

__version__ = get_distribution(__name__).version

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2018, zfit"
__status__ = "Beta"

__author__ = "Jonas Eschle"
__maintainer__ = "zfit"
__email__ = 'zfit@physik.uzh.ch'
__credits__ = ["Jonas Eschle <Jonas.Eschle@cern.ch>",
               "Albert Puig <apuignav@gmail.com",
               "Rafael Silva Coutinho <rafael.silva.coutinho@cern.ch>", ]

__all__ = ["ztf", "z", "constraint", "pdf", "minimize", "loss", "core", "data", "func", "dimension", "exception",
           "sample",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]


#  Copyright (c) 2019 zfit

# msg = inspect.cleandoc(
#     """zfit has moved from TensorFlow 1.x to 2.x, which has some profound
#     implications behind the scenes of zfit and minor ones on the user side.
#     Be sure to read the upgrade guide (can be found in the README at the top)
#     to have a seamless transition. If this is currently not doable you can
#     downgrade zfit to <0.4.
#     Feel free to contact us in case of problems in order to fix them ASAP.
#     """
# )
# warnings.warn(msg, stacklevel=2)


def _maybe_disable_warnings():
    import os
    if not os.environ.get("ZFIT_DISABLE_TF_WARNINGS"):
        return
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


_maybe_disable_warnings()

import tensorflow as tf

if int(tf.__version__[0]) < 2:
    raise RuntimeError(f"You are using TensorFlow version {tf.__version__}. This zfit version ({__version__}) works"
                       f" only with TF >= 2")

from . import z  # legacy
from . import (constraint, core, data, dimension, exception, func, loss,
               minimize, param, pdf, sample)
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
