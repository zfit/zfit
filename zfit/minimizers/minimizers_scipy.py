import tensorflow as tf

from zfit.core.minimizer import AbstractMinimizer
from zfit.minimizers.base_tf import AdapterTFOptimizer


class ScipyMinimizer(AdapterTFOptimizer, tf.contrib.opt.ScipyOptimizerInterface, AbstractMinimizer):
    pass
