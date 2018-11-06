import tensorflow as tf

from zfit.core.minimizer import MinimizerInterface
from zfit.minimizers.base_tf import AdapterTFOptimizer, WrapOptimizer


class AdadeltaMinimizer(AdapterTFOptimizer, tf.train.AdadeltaOptimizer, MinimizerInterface):
    pass


class AdagradMinimizer(AdapterTFOptimizer, tf.train.AdagradOptimizer, MinimizerInterface):
    pass


class GradientDescentMinimizer(AdapterTFOptimizer, tf.train.GradientDescentOptimizer,
                               MinimizerInterface):
    pass


class RMSPropMinimizer(AdapterTFOptimizer, tf.train.RMSPropOptimizer, MinimizerInterface):
    pass


class AdamMinimizer(AdapterTFOptimizer, tf.train.AdamOptimizer, MinimizerInterface):
    pass
