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
    # temp added
    "ScipyDogleg",
    "ScipyCOBYLA",
    "ScipyTrustNCG",
    "ScipyTrustKrylov",
    # temp added end
    "WrapOptimizer",
    "Adam",
    "Minuit",
    "LevenbergMarquardt",
    "ScipyBaseMinimizerV1",
    "ScipyLBFGSBV1",
    "ScipyTrustConstrV1",
    "ScipyPowellV1",
    "ScipySLSQPV1",
    "ScipyNewtonCGV1",
    "ScipyTruncNCV1",
    "ScipyNelderMeadV1",
    "ScipyBaseMinimizer",
    "ScipyLBFGSB",
    "ScipyTrustConstr",
    "ScipyPowell",
    "ScipySLSQP",
    "ScipyNewtonCG",
    "ScipyTruncNC",
    "ScipyNelderMead",
    "ScipyBFGS",
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
    "NLoptBaseMinimizer",
    "NLoptLBFGS",
    "NLoptTruncNewton",
    "NLoptSLSQP",
    "NLoptMMA",
    "NLoptCCSAQ",
    "NLoptShiftVar",
    "NLoptMLSL",
    "NLoptStoGO",
    "NLoptESCH",
    "NLoptISRES",
    "NLoptSubplex",
    "NLoptBOBYQA",
    "NLoptCOBYLA",
    "Ipyopt",
    "LevenbergMarquardt",
    "BaseMinimizer",
    "minimize_supports",
    "DefaultStrategy",
    "DefaultToyStrategy",
    "PushbackStrategy",
    "EDM",
]
