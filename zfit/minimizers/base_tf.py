#  Copyright (c) 2024 zfit

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

from collections.abc import Iterable

import tensorflow as tf

from ..core.interfaces import ZfitIndependentParameter, ZfitLoss
from ..util.exception import OperationNotAllowedError
from .baseminimizer import BaseStepMinimizer, minimize_supports


class WrapOptimizer(BaseStepMinimizer):
    def __init__(
        self,
        optimizer,
        tol=None,
        criterion=None,
        strategy=None,
        verbosity=None,
        name=None,
        **kwargs,
    ):
        """Wrap TensorFlow optimizers to have the zfit interface and behavior.

        .. warning:: The optimizers changed the interface in Keras 3, which is not (yet) supported.

          As they tend to perform significantly worse than the optimizers in `scipy` or `iminuit`, it is
            not a priority to support them, if ever at all. If you need them, please open an issue/PR: https://github.com/zfit/zfit.

        .. note:: Different behavior of minimize

          While the ``minimize`` method in TensorFlow optimizers executes a single step of the minimization,
          the ``minimize`` method of a :class:`ZfitMinimizer` fully minimizes a function until convergence
          is reached. In order to execute a single step, use the :meth:`~WrapOptimizer.step` method (however,
          this is in general not necessary to do and rather inefficient).

        Args:
            optimizer: Instance of a :class:`tf.keras.optimizers.Optimizer`.
            All other arguments: these are passed through and have the same functionality as described
            in :py:class:`~BaseStepMinimizer`
        """
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            msg = f"optimizer {optimizer!s} has to be from class Optimizer"
            raise TypeError(msg)
        if not hasattr(optimizer, "minimize"):
            msg = (
                f"optimizer {optimizer!s} has to have a method minimize and is most probably the new Keras 3 interface that is not yet supported."
                "Please open an issue/PR if you would need TF/Keras optimizers, as they generally don't perform well anyway."
            )
            raise AttributeError(msg)
        super().__init__(
            tol=tol,
            criterion=criterion,
            strategy=strategy,
            verbosity=verbosity,
            name=name,
            minimizer_options=None,
            **kwargs,
        )
        self._optimizer_tf = optimizer

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init):
        try:
            return super()._minimize(loss, params, init)
        except ValueError as error:
            if "No gradients provided for any variable" in error.args[0]:
                msg = "Cannot use TF optimizer with" " a numerical gradient (non-TF function)"
                raise OperationNotAllowedError(msg) from None
            raise

    def _step(
        self,
        loss: ZfitLoss,
        params: Iterable[ZfitIndependentParameter],
        init: zfit.result.FitResult | None,
    ) -> tf.Tensor:
        del init  # unused
        # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
        # import zfit
        # params = zfit.z.math._extract_tfparams(params)
        value = partial(loss.value, full=False)
        self._optimizer_tf.minimize(loss=value, var_list=params)
        return value()
