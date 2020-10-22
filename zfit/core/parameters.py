#  Copyright (c) 2020 zfit
from typing import Optional, Callable

import tensorflow as tf
import numpy as np
from tensorflow import Variable
from tensorflow.python.types.core import Tensor as TensorType

from .interfaces import ZfitIndependentParameter
from .parameter import ZfitParameterMixin
from .tensorlike import MetaBaseParameter, OverloadableMixin
from .. import z
from ..util import ztyping
from ..util.exception import NameAlreadyTakenError, IllegalInGraphModeError, ShapeIncompatibleError, WorkInProgressError


class BaseParameterArray(Variable, ZfitIndependentParameter, TensorType, metaclass=MetaBaseParameter):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except NotImplementedError:
            tmp_val = kwargs.pop('name', None)  # remove if name is in there, needs to be passed through
            if args or kwargs:
                kwargs['name'] = tmp_val
                raise RuntimeError(f"The following arguments reached the top of the inheritance tree, the super "
                                   f"init is not implemented (most probably abstract tf.Variable): {args, kwargs}. "
                                   f"If you see this error, please post it as an bug at: "
                                   f"https://github.com/zfit/zfit/issues/new/choose")


class ParameterArray(BaseParameterArray):

    def __init__(self, names, values, lower, upper, step_size=None, floating=None):
        from zfit import run
        run.assert_executing_eagerly()
        if not isinstance(names, (list, tuple)):
            raise TypeError("`name` has to be a list of names.")
        existing_names = set(self._existing_names).intersection(names)
        if existing_names:
            raise NameAlreadyTakenError("Another parameter is already named {}. "
                                        "Use a different, unique one.".format(existing_names))
        self._names = names = names
        size = int(tf.size(values))
        raise WorkInProgressError("TODO CONTINUE FROM HERE")
        if lower is not None and not values.shape == lower.shape:
            raise ShapeIncompatibleError

        if not values.shape == upper.shape:
            raise ShapeIncompatibleError
        super().__init__(name=f"Daddy_{names[0]}")

    # property needed here to overwrite the name of tf.Variable
    @property
    def name(self) -> str:
        return self._name

    def _set_single_value(self, value, index):
        own_index = tf.unravel_index(indices=index, dims=self.shape)
        self[own_index] = value


class ParameterArrayChild(ZfitParameterMixin, ZfitIndependentParameter, OverloadableMixin, TensorType,
                          metaclass=MetaBaseParameter):

    def __init__(self, index: int, paramarray: ParameterArray):
        name = paramarray.names[index]
        self._index = index
        self._independent = True
        super().__init__(name, dtype=paramarray.dtype)
        self._paramarray = paramarray

    def _extract_from_daddy(self, value):
        if isinstance(value, TensorType):
            return tf.reshape(value, (-1,))[self._index]
        else:  # assuming it's a Python collection
            return value[self._index]  # TODO: needed? which format?

    def value(self) -> tf.Tensor:
        return self._extract_from_daddy(self._paramarray.value())

    def set_value(self, value):
        super()._set_single_value(value=value, index=self._index)

    @property
    def independent(self) -> bool:
        return self._independent

    @property
    def step_size(self) -> tf.Tensor:
        return self._extract_from_daddy(self._paramarray.step_size)

    def read_value(self) -> tf.Tensor:
        return self.value()

    @property
    def lower(self):
        return self._extract_from_daddy(self._paramarray.lower)

    @property
    def upper(self):
        return self._extract_from_daddy(self._paramarray.upper)

    def _get_dependencies(self) -> ztyping.DependentsType:
        return {self}

    @property
    def has_limits(self) -> bool:
        return self.lower is not None and self.upper is not None  # TODO: improve, factor out?

    @property
    def at_limit(self) -> tf.Tensor:
        """If the value is at the limit (or over it).

        The precision is up to 1e-5 relative.

        Returns:
            Boolean `tf.Tensor` that tells whether the value is at the limits.
        """
        if not self.has_limits:
            return tf.constant(False)

        # Adding a slight tolerance to make sure we're not tricked by numerics
        at_lower = z.unstable.less_equal(self.value(), self.lower + (tf.math.abs(self.lower * 1e-5)))
        at_upper = z.unstable.greater_equal(self.value(), self.upper - (tf.math.abs(self.upper * 1e-5)))
        return z.unstable.logical_or(at_lower, at_upper)

    @property
    def shape(self):
        return tf.TensorShape([])

    @property
    def floating(self) -> bool:
        return self._extract_from_daddy(self._paramarray.floating)

    # TODO: factor out with Parameter
    def randomize(self, minval: Optional[ztyping.NumericalScalarType] = None,
                  maxval: Optional[ztyping.NumericalScalarType] = None,
                  sampler: Callable = np.random.uniform) -> tf.Tensor:
        """Update the parameter with a randomised value between minval and maxval and return it.


        Args:
            minval: The lower bound of the sampler. If not given, `lower_limit` is used.
            maxval: The upper bound of the sampler. If not given, `upper_limit` is used.
            sampler: A sampler with the same interface as `np.random.uniform`

        Returns:
            The sampled value
        """
        if not tf.executing_eagerly():
            raise IllegalInGraphModeError("Randomizing values in a parameter within Graph mode is most probably not"
                                          " what is wanted.")
        if minval is None:
            minval = self.lower
        else:
            minval = tf.cast(minval, dtype=self.dtype)
        if maxval is None:
            maxval = self.upper
        else:
            maxval = tf.cast(maxval, dtype=self.dtype)
        if maxval is None or minval is None:
            raise RuntimeError("Cannot randomize a parameter without limits or limits given.")
        value = sampler(size=self.shape, low=minval, high=maxval)

        self.set_value(value=value)
        return value
