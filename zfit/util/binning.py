#  Copyright (c) 2024 zfit
from __future__ import annotations

import tensorflow as tf

import zfit.z.numpy as znp

from ..settings import ztypes


def generate_1d_grid(data, num_grid_points, absolute_boundary=0.0, relative_boundary=0.05):
    absolute_boundary = znp.asarray(absolute_boundary, ztypes.float)
    relative_boundary = znp.asarray(relative_boundary, ztypes.float)
    num_grid_points = znp.asarray(num_grid_points, ztypes.int)
    minimum = znp.min(data)
    maximum = znp.max(data)
    space_width = maximum - minimum
    outside_borders = znp.maximum(relative_boundary * space_width, absolute_boundary)

    return tf.linspace(
        minimum - outside_borders, maximum + outside_borders, num=num_grid_points
    )  # requires tf, znp doesn't work
    # znp requires num_grid_points to be a python int and then can't do the division (as the others are floats)


def bin_1d(binning_method, data, grid, weights=None):
    if binning_method == "simple":
        return bin_1d_simple(data, grid, weights)
    elif binning_method == "linear":
        return bin_1d_linear(data, grid, weights)
    else:
        msg = f"Binning method '{binning_method}' not supported, only 'simple' or 'linear'."
        raise ValueError(msg)


def bin_1d_simple(data, grid, weights=None):
    if weights is None:
        bincount = znp.asarray(
            tf.histogram_fixed_width(
                data,
                [znp.min(grid), znp.max(grid)],
                znp.size(grid),
            ),
            ztypes.float,
        )
    else:
        bincount = _bin_1d_weighted(data, grid, weights, "simple")

    return bincount


def bin_1d_linear(data, grid, weights=None):
    return _bin_1d_weighted(data, grid, weights)


def _bin_1d_weighted(data, grid, weights, method="linear"):
    if weights is None:
        weights = znp.ones_like(data, ztypes.float)

    weights = weights / znp.sum(weights)

    grid_size = znp.size(grid)
    grid_min = znp.min(grid)
    grid_max = znp.max(grid)
    num_intervals = grid_size - tf.constant(1)
    dx = (grid_max - grid_min) / znp.asarray(num_intervals, ztypes.float)

    transformed_data = (data - grid_min) / dx

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    integral = znp.floor(transformed_data)
    fractional = transformed_data - integral

    if method == "simple":
        fractional = znp.asarray(fractional > 0.5, fractional.dtype) * fractional

    # Compute the weights for left and right side of the linear binning routine
    frac_weights = fractional * weights
    neg_frac_weights = weights - frac_weights

    # tf.math.bincount only works with tf.int32
    bincount_left = tf.roll(
        tf.concat(
            tf.math.bincount(
                znp.asarray(integral, tf.int32),
                weights=frac_weights,
                minlength=grid_size,
                maxlength=grid_size,
            ),
            tf.constant(0),
        ),
        shift=1,
        axis=0,
    )
    bincount_right = tf.math.bincount(
        znp.asarray(integral, tf.int32),
        weights=neg_frac_weights,
        minlength=grid_size,
        maxlength=grid_size,
    )

    return znp.asarray(bincount_left + bincount_right, ztypes.float)
