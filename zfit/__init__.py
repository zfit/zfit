# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

#  Copyright (c) 2020 zfit
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

__all__ = ["ztf", "z", "constraint", "pdf", "minimize", "loss", "core", "data", "func",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]

#  Copyright (c) 2019 zfit
warnings.warn(
    """zfit has moved from TensorFlow 1.x to 2.x, which has some profound implications behind the scenes of zfit
    and minor ones on the user side. Be sure to read the upgrade guide (can be found in the README at the top)
     to have a seemless transition. If this is currently not doable (upgrading is highly recommended though)
     you can downgrade zfit to <0.4. Feel free to contact us in case of problems in order to fix them ASAP.""")


def _maybe_disable_warnings():
    import os
    if not os.environ.get("ZFIT_DISABLE_TF_WARNINGS"):
        return
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # warnings.simplefilter(action='ignore', category=DeprecationWarning)
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


_maybe_disable_warnings()
import tensorflow.compat.v1 as _tfv1

_tfv1.enable_v2_behavior()
import tensorflow as tf

if int(tf.__version__[0]) < 2:
    warnings.warn(f"You are using TensorFlow version {tf.__version__}. This zfit version ({__version__}) works"
                  f" with TF >= 2 and will likely break with an older version. Please consider upgrading as this"
                  f" will raise an error in the future.")

# EXPERIMENTAL_FUNCTIONS_RUN_EAGERLY = False
# tf.config.experimental_run_functions_eagerly(EXPERIMENTAL_FUNCTIONS_RUN_EAGERLY)

from . import z
from . import z as ztf  # legacy
from .settings import ztypes

from . import constraint, pdf, minimize, loss, core, data, func, param
from .core.parameter import Parameter, ComposedParameter, ComplexParameter, convert_to_parameter
from .core.limits import Space, convert_to_space, supports
from .core.data import Data

from .settings import run


def _maybe_disable_jit():
    import os
    z.zextension.FunctionWrapperRegistry.do_jit = bool(int(os.environ.get("ZFIT_DO_JIT", True)))


_maybe_disable_jit()

# experimental flags
experimental_loss_penalty_nan = False

# EOF
