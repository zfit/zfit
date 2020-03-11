#  Copyright (c) 2020 zfit
try:
    from tensorflow.python import deprecated
except ImportError:  # TF < 2.2
    from tensorflow_core.python import deprecated

from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_tfp import BFGS
from .minimizers.minimizers_scipy import Scipy
from .minimizers.optimizers_tf import Adam, WrapOptimizer


@deprecated(None, "Use zfit.minimize.Adam instead.")
class AdamMinimizer(Adam):
    pass


@deprecated(None, "Use zfit.minimize.Minuit instead.")
class MinuitMinimizer(Minuit):
    pass


@deprecated(None, "Use zfit.minimize.Scipy instead.")
class ScipyMinimizer(Scipy):
    pass


__all__ = ['MinuitMinimizer', 'ScipyMinimizer', 'AdamMinimizer',
           "WrapOptimizer",
           "Adam", "Minuit", "Scipy", "BFGS"]
