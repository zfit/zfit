#  Copyright (c) 2021 zfit


from .minimizers.baseminimizer import DefaultStrategy, DefaultToyStrategy, ZfitStrategy
from .minimizers.fitresult import FitResult
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_nlopt import (NLopt, NLoptLBFGSV1, NLoptTruncNewtonV1, NLoptSLSQPV1, NLoptMMAV1,
                                         NLoptCCSAQV1,
                                         NLoptSubplexV1,
    # NLoptMLSLV1,
                                         )
from .minimizers.minimizer_tfp import BFGS
from .minimizers.minimizers_scipy import Scipy, ScipyLBFGSBV1, ScipyTrustKrylovV1, ScipyTrustConstrV1, ScipyDoglegV1, \
    ScipyTrustNCGV1
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .util.legacy import deprecated


class AdamMinimizer(Adam):

    @deprecated(None, "Use zfit.minimize.Adam instead.")
    def __init__(self, tolerance=None, learning_rate=0.2, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam', **kwargs):
        super().__init__(tolerance, learning_rate, beta1, beta2, epsilon, name, **kwargs)


class MinuitMinimizer(Minuit):

    @deprecated(None, "Use zfit.minimize.Minuit instead.")
    def __init__(self, strategy: ZfitStrategy = None, minimize_strategy: int = 1, tolerance: float = None,
                 verbosity: int = 5, name: str = None, ncall: int = 10000, use_minuit_grad: bool = None,
                 **minimizer_options):
        super().__init__(strategy, minimize_strategy, tolerance, verbosity, name, ncall, use_minuit_grad,
                         **minimizer_options)


class ScipyMinimizer(Scipy):

    @deprecated(None, "Use zfit.minimize.Scipy instead.")
    def __init__(self, minimizer='L-BFGS-B', tolerance=None, verbosity=5, name=None, **minimizer_options):
        super().__init__(minimizer, tolerance, verbosity, name, **minimizer_options)


__all__ = ['MinuitMinimizer', 'ScipyMinimizer', 'AdamMinimizer',
           "WrapOptimizer",
           "Adam", "Minuit",
           "Scipy", "ScipyLBFGSBV1", "ScipyTrustKrylovV1", 'ScipyTrustConstrV1', "ScipyDoglegV1", "ScipyTrustNCGV1",
           "NLoptLBFGSV1", "NLoptTruncNewtonV1", "NLoptSLSQPV1", "NLoptMMAV1", "NLoptCCSAQV1",
           # "NLoptMLSLV1",
           "NLoptSubplexV1",
           "BFGS", "NLopt",
           "DefaultStrategy", "DefaultToyStrategy",
           "FitResult"]
