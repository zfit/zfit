#  Copyright (c) 2024 zfit
from __future__ import annotations

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
from .minimizers.strategy import DefaultToyStrategy, PushbackStrategy
from .minimizers.termination import EDM

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
