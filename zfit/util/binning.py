#  Copyright (c) 2022 zfit

import tensorflow as tf

from ..settings import ztypes


def generate_1d_grid(
    data, num_grid_points, absolute_boundary=0.0, relative_boundary=0.05
):
    minimum = tf.math.reduce_min(data)
    maximum = tf.math.reduce_max(data)
    space_width = maximum - minimum
    outside_borders = tf.maximum(relative_boundary * space_width, absolute_boundary)

    return tf.linspace(
        minimum - outside_borders, maximum + outside_borders, num=num_grid_points
    )


def bin_1d(binning_method, data, grid, weights=None):
    if binning_method == "simple":
        return bin_1d_simple(data, grid, weights)
    elif binning_method == "linear":
        return bin_1d_linear(data, grid, weights)
    else:
        raise ValueError(
            f"Binning method '{binning_method}' not supported, only 'simple' or 'linear'."
        )


def bin_1d_simple(data, grid, weights=None):
    if weights is None:
        bincount = tf.cast(
            tf.histogram_fixed_width(
                data,
                [tf.math.reduce_min(grid), tf.math.reduce_max(grid)],
                tf.size(grid),
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
        weights = tf.ones_like(data, ztypes.float)

    weights = weights / tf.reduce_sum(weights)

    grid_size = tf.size(grid)
    grid_min = tf.math.reduce_min(grid)
    grid_max = tf.math.reduce_max(grid)
    num_intervals = tf.math.subtract(grid_size, tf.constant(1))
    dx = tf.math.divide(
        tf.math.subtract(grid_max, grid_min), tf.cast(num_intervals, ztypes.float)
    )

    transformed_data = tf.math.divide(tf.math.subtract(data, grid_min), dx)

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    integral = tf.math.floor(transformed_data)
    fractional = tf.math.subtract(transformed_data, integral)

    if method == "simple":
        fractional = tf.cast(fractional > 0.5, fractional.dtype) * fractional

    # Compute the weights for left and right side of the linear binning routine
    frac_weights = tf.math.multiply(fractional, weights)
    neg_frac_weights = tf.math.subtract(weights, frac_weights)

    # tf.math.bincount only works with tf.int32
    bincount_left = tf.roll(
        tf.concat(
            tf.math.bincount(
                tf.cast(integral, tf.int32),
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
        tf.cast(integral, tf.int32),
        weights=neg_frac_weights,
        minlength=grid_size,
        maxlength=grid_size,
    )

    bincount = tf.cast(tf.add(bincount_left, bincount_right), ztypes.float)

    return bincount
