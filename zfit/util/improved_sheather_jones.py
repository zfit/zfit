#  Copyright (c) 2023 zfit

import numpy as np
import tensorflow as tf

from . import binning as binning_util
from .. import z
from ..settings import ztypes
from . import root_search


@z.function(wraps="tensor")
def calc_f(s, f, squared_integers, grid_data_dct2, N):
    # Step one: estimate t_s from |f^(s+1)|^2
    one_half = tf.constant(1.0 / 2.0, ztypes.float)
    one = tf.constant(1.0, ztypes.float)
    two = tf.constant(2.0, ztypes.float)
    three = tf.constant(3.0, ztypes.float)
    pi = tf.constant(np.pi, ztypes.float)

    odd_numbers_prod = tf.math.reduce_prod(tf.range(one, two * s + one, 2))
    K0 = odd_numbers_prod / tf.math.sqrt(two * pi)
    const = (one + tf.math.pow(one_half, s + one_half)) / three
    time = tf.math.pow(two * const * K0 / (N * f), two / (three + two * s))

    # Step two: estimate |f^s| from t_s
    f = (
        one_half
        * tf.math.pow(pi, (two * s))
        * tf.math.reduce_sum(
            tf.math.pow(squared_integers, s)
            * grid_data_dct2
            * tf.math.exp(-squared_integers * tf.math.pow(pi, two) * time)
        )
    )

    return f


@z.function(wraps="tensor", autograph=True)
def _fixed_point(t, N, squared_integers, grid_data_dct2):
    r"""Compute the fixed point as described in the paper by Botev et al.

    .. math:
        t = \xi \gamma^{5}(t)

    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    squared_integers : array-like
        The numbers [1, 2, 9, 16, ...]
    grid_data_dct2 : array-like
        The DCT of the original data, divided by 2 and squared.
    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01, 50, np.arange(1, 51), np.arange(1, 51))
    >>> assert np.allclose(ans, 0.0099076220293967618515)
    >>> # another
    >>> ans = _fixed_point(0.07, 25, np.arange(1, 11), np.arange(1, 11))
    >>> assert np.allclose(ans, 0.068342291525717486795)
    References
    ----------
     - Implementation by Daniel B. Smith, PhD, found at
       https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    """
    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = tf.constant(7, ztypes.float)

    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = (
        tf.constant(0.5, ztypes.float)
        * tf.math.pow(
            tf.constant(np.pi, ztypes.float), (tf.constant(2.0, ztypes.float) * ell)
        )
        * tf.math.reduce_sum(
            tf.math.pow(squared_integers, ell)
            * grid_data_dct2
            * tf.math.exp(
                -squared_integers
                * tf.math.pow(
                    tf.constant(np.pi, ztypes.float), tf.constant(2.0, ztypes.float)
                )
                * t
            )
        )
    )

    i = tf.constant(6.0, dtype=ztypes.float)
    while_condition = lambda i, f: i > 1

    def body(i, f):
        # do something here which you want to do in your loop
        # increment i
        f = calc_f(i, f, squared_integers, grid_data_dct2, N)
        return i - 1.0, f

    # do the loop:
    fnew = tf.while_loop(
        while_condition, body, (i, f), maximum_iterations=5, parallel_iterations=5
    )[1]

    # This is the minimizer of the AMISE
    t_opt = tf.math.pow(
        tf.constant(2 * np.sqrt(np.pi), ztypes.float) * N * fnew,
        tf.constant(-2.0 / 5.0, ztypes.float),
    )

    # Return the difference between the original t and the optimal value
    return t - t_opt


def _find_root(function, N, squared_integers, grid_data_dct2):
    """Root finding algorithm. Based on MATLAB implementation by Botev et al.

    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> np.allclose(ans, 9.237610787616029e-05)
    True
    """

    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N2 = tf.math.maximum(
        tf.math.minimum(tf.constant(1050, ztypes.float), N),
        tf.constant(50, ztypes.float),
    )
    tol = 10e-12 + 0.01 * (N2 - 50) / 1000
    left_bracket = tf.constant(0.0, dtype=ztypes.float)
    right_bracket = tf.constant(10e-12, ztypes.float) + tf.constant(
        0.01, ztypes.float
    ) * (N2 - tf.constant(50, ztypes.float)) / tf.constant(1000, ztypes.float)

    converged = tf.constant(False)
    t_star = tf.constant(0.0, dtype=ztypes.float)

    def fixed_point_function(t):
        return _fixed_point(t, N, squared_integers, grid_data_dct2)

    def condition(right_bracket, converged, t_star):
        return tf.math.logical_not(converged)

    def body(right_bracket, converged, t_star):
        t_star, value_at_t_star, num_iterations, converged = root_search.brentq(
            fixed_point_function, left_bracket, right_bracket, None, None, 2e-12
        )

        t_star = t_star - value_at_t_star

        right_bracket = right_bracket * tf.constant(2.0, ztypes.float)

        return right_bracket, converged, t_star

    # While a solution is not found, increase the tolerance and try again
    right_bracket, converged, t_star = tf.while_loop(
        condition, body, [right_bracket, converged, t_star]
    )

    return t_star


def _calculate_t_star(data, num_grid_points, binning_method, weights):
    # Setting `percentile` higher decreases the chance of overflow
    grid = binning_util.generate_1d_grid(data, num_grid_points, 6.0, 0.5)

    # Create an equidistant grid
    R = tf.cast(tf.reduce_max(data) - tf.reduce_min(data), ztypes.float)

    # dx = R / tf.constant((num_grid_points - 1), ztypes.float)
    data_unique, data_unique_indexes = tf.unique(data)
    N = tf.cast(tf.size(data_unique), ztypes.float)

    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    grid_data = binning_util.bin_1d(binning_method, data, grid, weights)

    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    grid_data_dct = tf.signal.dct(grid_data, type=2)

    # Compute the bandwidth
    squared_integers = tf.math.pow(
        tf.range(1, num_grid_points, dtype=ztypes.float), tf.constant(2, ztypes.float)
    )
    grid_data_dct2 = tf.math.pow(grid_data_dct[1:], 2) / 4

    # Solve for the optimal (in the AMISE sense) t
    t_star = _find_root(_fixed_point, N, squared_integers, grid_data_dct2)

    return t_star, R, squared_integers, grid_data_dct, grid


def _calculate_density(t_star, R, squared_integers, grid_data_dct):
    # Prepend zero
    squared_integers = tf.pad(squared_integers, [[1, 0]])

    # Smooth the initial data using the computed optimal t
    # Multiplication in frequency domain is convolution
    grid_data_dct_t = grid_data_dct * tf.math.exp(
        -squared_integers
        * tf.math.pow(tf.constant(np.pi, ztypes.float), tf.constant(2.0, ztypes.float))
        * t_star
        / tf.constant(2.0, ztypes.float)
    )

    # Diving by 2 done because of the implementation of tf.signal.idct
    density = tf.signal.idct(grid_data_dct_t, type=2) / (2 * R)

    # Due to overflow, some values might be smaller than zero, correct it
    density = tf.cast(density > 0, density.dtype) * density

    return density


def calculate_bandwidth(
    data, num_grid_points=1024, binning_method="linear", weights=None
):
    data = tf.cast(data, ztypes.float)

    t_star, R, squared_integers, grid_data_dct, grid = _calculate_t_star(
        data, num_grid_points, binning_method, weights
    )

    return tf.math.sqrt(t_star) * R


def calculate_bandwidth_and_density(
    data, num_grid_points=1024, binning_method="linear", weights=None
):
    data = tf.cast(data, ztypes.float)

    t_star, R, squared_integers, grid_data_dct, grid = _calculate_t_star(
        data, num_grid_points, binning_method, weights
    )

    bandwidth = tf.math.sqrt(t_star) * R
    density = _calculate_density(t_star, R, squared_integers, grid_data_dct)

    return bandwidth, density, grid
