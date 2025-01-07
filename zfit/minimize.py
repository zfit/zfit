#  Copyright (c) 2024 zfit
from __future__ import annotations

from .minimizers.baseminimizer import (
    BaseMinimizer,
    DefaultStrategy,
    minimize_supports,
)
from .minimizers.ipopt import Ipyopt
from .minimizers.minimizer_lm import LevenbergMarquardt
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizer_nlopt import (
    NLoptBaseMinimizer,
    NLoptBOBYQA,
    NLoptCCSAQ,
    NLoptCOBYLA,
    NLoptESCH,
    NLoptISRES,
    NLoptLBFGS,
    NLoptMLSL,
    NLoptMMA,
    NLoptShiftVar,
    NLoptSLSQP,
    NLoptStoGO,
    NLoptSubplex,
    NLoptTruncNewton,
)
from .minimizers.minimizers_scipy import (
    ScipyBaseMinimizer,
    ScipyBFGS,
    ScipyCOBYLA,
    ScipyDogleg,
    ScipyLBFGSB,
    ScipyNelderMead,
    ScipyNewtonCG,
    ScipyPowell,
    ScipySLSQP,
    ScipyTruncNC,
    ScipyTrustConstr,
    ScipyTrustKrylov,
    ScipyTrustNCG,
)
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .minimizers.strategy import DefaultToyStrategy, PushbackStrategy
from .minimizers.termination import EDM

ScipyTrustConstrV1 = ScipyTrustConstr
ScipyTrustNCGV1 = ScipyTrustNCG
ScipyTrustKrylovV1 = ScipyTrustKrylov
ScipyDoglegV1 = ScipyDogleg
ScipyCOBYLAV1 = ScipyCOBYLA
ScipyLBFGSBV1 = ScipyLBFGSB
ScipyPowellV1 = ScipyPowell
ScipySLSQPV1 = ScipySLSQP
ScipyNewtonCGV1 = ScipyNewtonCG
ScipyTruncNCV1 = ScipyTruncNC
ScipyNelderMeadV1 = ScipyNelderMead
NLoptLBFGSV1 = NLoptLBFGS
NLoptTruncNewtonV1 = NLoptTruncNewton
NLoptSLSQPV1 = NLoptSLSQP
NLoptMMAV1 = NLoptMMA
NLoptCCSAQV1 = NLoptCCSAQ
NLoptShiftVarV1 = NLoptShiftVar
NLoptMLSLV1 = NLoptMLSL
NLoptStoGOV1 = NLoptStoGO
NLoptESCHV1 = NLoptESCH
NLoptISRESV1 = NLoptISRES
NLoptSubplexV1 = NLoptSubplex
NLoptBOBYQAV1 = NLoptBOBYQA
NLoptCOBYLAV1 = NLoptCOBYLA
IpyoptV1 = Ipyopt
ScipyBaseMinimizerV1 = ScipyBaseMinimizer
NLoptBaseMinimizerV1 = NLoptBaseMinimizer

BaseMinimizerV1 = BaseMinimizer


__all__ = [
    "EDM",
    "Adam",
    "BaseMinimizer",
    "DefaultStrategy",
    "DefaultToyStrategy",
    "Ipyopt",
    "LevenbergMarquardt",
    "LevenbergMarquardt",
    "Minuit",
    "NLoptBOBYQA",
    "NLoptBOBYQAV1",
    "NLoptBaseMinimizer",
    "NLoptBaseMinimizerV1",
    "NLoptCCSAQ",
    "NLoptCCSAQV1",
    "NLoptCOBYLA",
    "NLoptCOBYLAV1",
    "NLoptESCH",
    "NLoptESCHV1",
    "NLoptISRES",
    "NLoptISRESV1",
    "NLoptLBFGS",
    "NLoptLBFGSV1",
    "NLoptMLSL",
    "NLoptMLSLV1",
    "NLoptMMA",
    "NLoptMMAV1",
    "NLoptSLSQP",
    "NLoptSLSQPV1",
    "NLoptShiftVar",
    "NLoptShiftVarV1",
    "NLoptStoGO",
    "NLoptStoGOV1",
    "NLoptSubplex",
    "NLoptSubplexV1",
    "NLoptTruncNewton",
    "NLoptTruncNewtonV1",
    "PushbackStrategy",
    "ScipyBFGS",
    "ScipyBaseMinimizer",
    "ScipyBaseMinimizerV1",
    "ScipyCOBYLA",
    # temp added
    "ScipyDogleg",
    "ScipyLBFGSB",
    "ScipyLBFGSBV1",
    "ScipyNelderMead",
    "ScipyNelderMeadV1",
    "ScipyNewtonCG",
    "ScipyNewtonCGV1",
    "ScipyPowell",
    "ScipyPowellV1",
    "ScipySLSQP",
    "ScipySLSQPV1",
    "ScipyTruncNC",
    "ScipyTruncNCV1",
    "ScipyTrustConstr",
    "ScipyTrustConstrV1",
    "ScipyTrustKrylov",
    "ScipyTrustNCG",
    # temp added end
    "WrapOptimizer",
    "minimize_supports",
]
