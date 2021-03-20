#  Copyright (c) 2021 zfit


from .minimizers.baseminimizer import DefaultStrategy
from .minimizers.fitresult import FitResult
from .minimizers.ipopt import IpyoptV1
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_nlopt import (NLoptBOBYQAV1, NLoptCCSAQV1,
                                         NLoptESCHV1, NLoptISRESV1,
                                         NLoptLBFGSV1, NLoptMLSLV1, NLoptMMAV1,
                                         NLoptShiftVarV1, NLoptSLSQPV1,
                                         NLoptStoGOV1, NLoptSubplexV1,
                                         NLoptTruncNewtonV1)
from .minimizers.minimizer_tfp import BFGS
from .minimizers.minimizers_scipy import (ScipyDoglegV1, ScipyLBFGSBV1,
                                          ScipyNewtonCGV1, ScipyPowellV1,
                                          ScipySLSQPV1, ScipyTruncNCV1,
                                          ScipyTrustConstrV1,
                                          ScipyTrustKrylovV1, ScipyTrustNCGV1)
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .minimizers.strategy import ZfitStrategy
from .util.deprecation import deprecated


class AdamMinimizer(Adam):

    @deprecated(None, "Use zfit.minimize.Adam instead.")
    def __init__(self, tol=None, learning_rate=0.2, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam', **kwargs):
        super().__init__(tol, learning_rate, beta1, beta2, epsilon, name, **kwargs)


class MinuitMinimizer(Minuit):

    @deprecated(None, "Use zfit.minimize.Minuit instead.")
    def __init__(self, strategy: ZfitStrategy = None, minimize_strategy: int = 1, tol: float = None,
                 verbosity: int = 5, name: str = None, ncall: int = 10000, use_minuit_grad: bool = None,
                 **options):
        super().__init__(strategy, minimize_strategy, tol, verbosity, name, ncall, use_minuit_grad,
                         **options)


class Scipy:
    def __init__(self, *_, **__):
        raise RuntimeError("This has been removed. Use the new Scipy* minimizer instead. In case anyone is missing,"
                           " please feel free to open a request:"
                           " https://github.com/zfit/zfit/issues/new?assignees="
                           "&labels=discussion&template=feature-request.md&title="
                           " or directly make a PR.")


class ScipyMinimizer(Scipy):
    pass


__all__ = ['MinuitMinimizer', 'ScipyMinimizer', 'AdamMinimizer',
           "WrapOptimizer",
           "Adam", "Minuit",
           "Scipy", "ScipyLBFGSBV1", "ScipyTrustKrylovV1", 'ScipyTrustConstrV1', "ScipyDoglegV1", "ScipyTrustNCGV1",
           "ScipyPowellV1", "ScipySLSQPV1", "ScipyNewtonCGV1", "ScipyTruncNCV1",
           "NLoptLBFGSV1", "NLoptTruncNewtonV1", "NLoptSLSQPV1", "NLoptMMAV1", "NLoptCCSAQV1", 'NLoptShiftVarV1',
           "NLoptMLSLV1", 'NLoptStoGOV1', 'NLoptESCHV1', 'NLoptISRESV1',
           "IpyoptV1",
           "NLoptSubplexV1", "NLoptBOBYQAV1",
           "BFGS",
           "DefaultStrategy", "FitResult"]
