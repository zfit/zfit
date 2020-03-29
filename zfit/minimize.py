#  Copyright (c) 2020 zfit


from .minimizers.baseminimizer import DefaultStrategy, DefaultToyStrategy
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_tfp import BFGS
from .minimizers.minimizers_scipy import Scipy
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .util.legacy import deprecated


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
           "Adam", "Minuit", "Scipy", "BFGS",
           "DefaultStrategy", "DefaultToyStrategy"]
