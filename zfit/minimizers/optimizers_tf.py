import tensorflow as tf

from zfit.core.minimizer import AbstractMinimizer
from zfit.minimizers.base_tf import AdapterTFOptimizer


class AdadeltaMinimizer(AdapterTFOptimizer, tf.train.AdadeltaOptimizer, AbstractMinimizer):
    pass


class AdagradMinimizer(AdapterTFOptimizer, tf.train.AdagradOptimizer, AbstractMinimizer):
    pass


class GradientDescentMinimizer(AdapterTFOptimizer, tf.train.GradientDescentOptimizer,
                               AbstractMinimizer):
    pass


class RMSPropMinimizer(AdapterTFOptimizer, tf.train.RMSPropOptimizer, AbstractMinimizer):
    pass


class AdamMinimizer(AdapterTFOptimizer, tf.train.AdamOptimizer, AbstractMinimizer):
    pass
