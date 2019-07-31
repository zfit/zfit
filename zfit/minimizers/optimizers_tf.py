#  Copyright (c) 2019 zfit

import tensorflow as tf

from .base_tf import WrapOptimizer


# class AdadeltaMinimizer(AdapterTFOptimizer, tf.train.AdadeltaOptimizer, ZfitMinimizer):
#     def __init__(self):
#         raise NotImplementedError("Currently a placeholder, has to be implemented (with WrapOptimizer")
#
#
# class AdagradMinimizer(AdapterTFOptimizer, tf.train.AdagradOptimizer, ZfitMinimizer):
#     def __init__(self):
#         raise NotImplementedError("Currently a placeholder, has to be implemented (with WrapOptimizer")
#
#
# class GradientDescentMinimizer(AdapterTFOptimizer, tf.train.GradientDescentOptimizer, ZfitMinimizer):
#     def __init__(self):
#         raise NotImplementedError("Currently a placeholder, has to be implemented (with WrapOptimizer")
#
#
# class RMSPropMinimizer(AdapterTFOptimizer, tf.train.RMSPropOptimizer, ZfitMinimizer):
#     def __init__(self):
#         raise NotImplementedError("Currently a placeholder, has to be implemented (with WrapOptimizer")


class Adam(WrapOptimizer):
    _DEFAULT_name = 'Adam'

    def __init__(self, tolerance=None,
                 learning_rate=0.2,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 use_locking=False,
                 name='Adam', **kwargs):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1, beta2=beta2,
                                           epsilon=epsilon, use_locking=use_locking,
                                           name=name)
        super().__init__(optimizer=optimizer, tolerance=tolerance, **kwargs)
