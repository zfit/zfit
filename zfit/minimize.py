#  Copyright (c) 2022 zfit


from .minimizers.baseminimizer import (
    BaseMinimizer,
    BaseMinimizerV1,
    DefaultStrategy,
    minimize_supports,
)
from .minimizers.ipopt import IpyoptV1
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_nlopt import (
    NLoptBaseMinimizerV1,
    NLoptBOBYQAV1,
    NLoptCCSAQV1,
    NLoptCOBYLAV1,
    NLoptESCHV1,
    NLoptISRESV1,
    NLoptLBFGSV1,
    NLoptMLSLV1,
    NLoptMMAV1,
    NLoptShiftVarV1,
    NLoptSLSQPV1,
    NLoptStoGOV1,
    NLoptSubplexV1,
    NLoptTruncNewtonV1,
)
from .minimizers.minimizers_scipy import (
    ScipyBaseMinimizerV1,
    ScipyLBFGSBV1,
    ScipyNelderMeadV1,
    ScipyNewtonCGV1,
    ScipyPowellV1,
    ScipySLSQPV1,
    ScipyTruncNCV1,
    ScipyTrustConstrV1,
)
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .minimizers.strategy import DefaultToyStrategy, PushbackStrategy, ZfitStrategy
from .minimizers.termination import EDM


class AdamMinimizer(Adam):
    def __init__(
        self,
        tol=None,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        name="Adam",
        **kwargs
    ):
        raise RuntimeError("Use zfit.minimize.Adam instead.")


class MinuitMinimizer(Minuit):
    def __init__(
        self,
        strategy: ZfitStrategy = None,
        minimize_strategy: int = 1,
        tol: float = None,
        verbosity: int = 5,
        name: str = None,
        ncall: int = 10000,
        use_minuit_grad: bool = None,
        **options
    ):
        raise RuntimeError(None, "Use zfit.minimize.Minuit instead.")


class Scipy:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "This has been removed. Use the new Scipy* minimizer instead. In case anyone is missing,"
            " please feel free to open a request:"
            " https://github.com/zfit/zfit/issues/new?assignees="
            "&labels=discussion&template=feature-request.md&title="
            " or directly make a PR."
        )


class ScipyMinimizer(Scipy):
    pass


class BFGS:
    def __init__(self) -> None:
        raise RuntimeError(
            "BFGS (from TensorFlow Probability) has been removed as it is currently"
            " not working well. Use other BFGS-like implementations such as ScipyLBFGSBV1"
            " or NLoptLBFGSV1."
        )


__all__ = [
    "WrapOptimizer",
    "Adam",
    "Minuit",
    "ScipyBaseMinimizerV1",
    "ScipyLBFGSBV1",
    "ScipyTrustConstrV1",
    "ScipyPowellV1",
    "ScipySLSQPV1",
    "ScipyNewtonCGV1",
    "ScipyTruncNCV1",
    "ScipyNelderMeadV1",
    "NLoptBaseMinimizerV1",
    "NLoptLBFGSV1",
    "NLoptTruncNewtonV1",
    "NLoptSLSQPV1",
    "NLoptMMAV1",
    "NLoptCCSAQV1",
    "NLoptShiftVarV1",
    "NLoptMLSLV1",
    "NLoptStoGOV1",
    "NLoptESCHV1",
    "NLoptISRESV1",
    "NLoptSubplexV1",
    "NLoptBOBYQAV1",
    "NLoptCOBYLAV1",
    "IpyoptV1",
    "BaseMinimizer",
    "BaseMinimizerV1",
    "minimize_supports",
    "DefaultStrategy",
    "DefaultToyStrategy",
    "PushbackStrategy",
    "EDM",
]
