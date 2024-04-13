#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from .util.execution import RunManager

run = RunManager()


def set_seed(seed=None, numpy=None, backend=None, zfit=None):
    """Set random seed for zfit, numpy and the backend. If seed is given, this is used to generate deterministic seeds
    for numpy, zfit and the backend.

    Notably, no python random seed is set.

    .. warning::

        To use this function to guarantee randomness, do *not* specify a seed.
        Use a seed _only_ for reproducibility: there have been unique cases of correlation between the seed and the result.
        As the seeds are returned, you can save them and rerun *if needed*.

    Uses `os.urandom` to generate a seed if `None` is given, which guarantees cryptographically secure randomness.

    Args:
        seed (int, optional): Seed to set for the random number generator of the seeds for within zfit. If `None` (default), the seed is set to a random value.
        numpy (int, bool, optional): Seed to set for numpy. If `True` (default), a random seed depending on the seed as used for zfit is used. If `False`, the seed is not set.
        backend (int, bool, optional): Seed to set for the backend. If `True` (default), a random seed depending on the seed as used for zfit is used. If `False`, the seed is not set.
        zfit (int, bool, optional): Seed to set for zfit. If `True` (default), a random seed is used. If `False`, the seed is not set.

    Returns:
        dict: Seeds that were set, with the keys `zfit`, `numpy`, `backend`.
    """
    if seed is None:
        seed = generate_urandom_seed()

    if numpy is None:
        numpy = True
    if backend is None:
        backend = True
    if zfit is None:
        zfit = True
    initial_seed = seed

    if numpy is True:
        if seed is True:
            numpy = generate_urandom_seed()
        else:
            rng = np.random.default_rng(seed)
            seed = rng.integers(0, 2**31 - 1)
            numpy = seed
    if backend is True:
        if seed is True:
            backend = generate_urandom_seed()
        else:
            rng = np.random.default_rng(seed)
            seed = rng.integers(0, 2**31 - 1)
            backend = seed

    if numpy is not None and numpy is not False:
        np.random.seed(numpy)
    if backend is not None and backend is not False:
        tf.random.set_seed(backend)

    from .z.random import get_prng

    if zfit is True:
        if seed is True:
            zfit = generate_urandom_seed()
        else:
            rng = np.random.default_rng(seed)
            seed = rng.integers(0, 2**31 - 1)
            zfit = seed
    if zfit is not None and zfit is not False:
        get_prng().reset_from_seed(zfit)
    seeds = {"zfit": zfit, "numpy": numpy, "backend": backend}
    if initial_seed is not True:
        seeds["seed"] = initial_seed
    return seeds


def generate_urandom_seed():
    import os

    random_data = os.urandom(4)
    backend_seed = int.from_bytes(random_data, byteorder="big")
    return int(abs(backend_seed) % 2**31)  # make sure it's positive and not too large


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
        "auto_update_params": True,
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
