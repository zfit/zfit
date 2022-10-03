#  Copyright (c) 2022 zfit
from __future__ import annotations

import tensorflow as tf

from .base_tf import WrapOptimizer


class Adam(WrapOptimizer):
    # todo see why sphinx does not correctly link to WrapOptimizer for api
    _DEFAULT_name = "Adam"

    def __init__(
        self,
        tol=None,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        name="Adam",
        **kwargs,
    ):
        # todo write documentation for api.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
            name=name,
        )
        super().__init__(optimizer=optimizer, tol=tol, maxiter=None, **kwargs)
