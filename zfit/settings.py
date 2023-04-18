#  Copyright (c) 2023 zfit

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from .util.execution import RunManager

run = RunManager()


def set_seed(seed=None, numpy=None, backend=None):
    """Set random seed for zfit, numpy and the backend.

    Args:
        seed (int, optional): Seed to set for the random number generator within zfit. If `None` (default), the seed is set to a random value.
        numpy (int, bool, optional): Seed to set for numpy. If `True` (default), the same seed as for zfit is used. If `False`, the seed is not set.
        backend (int, bool, optional): Seed to set for the backend. If `True` (default), the same seed as for zfit is used. If `False`, the seed is not set.
    """
    if seed is None:
        import os

        random_data = os.urandom(4)
        backend_seed = int.from_bytes(random_data, byteorder="big")
        backend_seed = int(
            abs(backend_seed) % 2**31
        )  # make sure it's positive and not too large
        seed = backend_seed

    if numpy is None:
        numpy = True
    if backend is None:
        backend = True

    if numpy is True:
        numpy = seed
    if backend is True:
        backend = seed

    if numpy is not None and numpy is not False:
        np.random.seed(numpy)
    if backend is not None and backend is not False:
        tf.random.set_seed(backend)

    from .z.random import get_prng

    get_prng().reset_from_seed(seed)


_verbosity = 0


def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity


def get_verbosity():
    return _verbosity


ztypes = DotMap(
    {
        "float": tf.float64,
        "complex": tf.complex128,
        "int": tf.int64,
        "auto_upcast": True,
    }
)
upcast_ztypes = {
    tf.float16: tf.float64,
    tf.float32: tf.float64,
    tf.float64: tf.float64,
    tf.complex64: tf.complex128,
    tf.complex128: tf.complex128,
    tf.int8: tf.int64,
    tf.int16: tf.int64,
    tf.int32: tf.int64,
    tf.int64: tf.int64,
}

options = DotMap(
    {
        "epsilon": 1e-8,
        "numerical_grad": None,
        "advanced_warning": True,
        "changed_warning": True,
    }
)

advanced_warnings = DotMap(
    {
        "sum_extended_frac": True,
        "exp_shift": True,
        "py_func_autograd": True,
        "inconsistent_fitrange": True,
        "extended_in_UnbinnedNLL": True,
        "all": True,
    }
)

changed_warnings = DotMap(
    {
        "new_sum": True,
        "all": True,
    }
)
