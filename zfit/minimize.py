#  Copyright (c) 2020 zfit

# from .minimizers.optimizers_tf import RMSPropMinimizer, GradientDescentMinimizer, AdagradMinimizer, AdadeltaMinimizer,
from .minimizers.optimizers_tf import Adam, WrapOptimizer
from .minimizers.minimizer_tfp import BFGS
from .minimizers.minimizer_minuit import Minuit
from .minimizers.minimizers_scipy import Scipy

AdamMinimizer = Adam  # legacy
MinuitMinimizer = Minuit  # legacy
ScipyMinimizer = Scipy  # legacy
__all__ = ['MinuitMinimizer', 'ScipyMinimizer', 'AdamMinimizer',
           "WrapOptimizer",
           "Adam", "Minuit", "Scipy", "BFGS"]
