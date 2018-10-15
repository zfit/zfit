from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from zfit.core.utils import dotdict


def auto_integrate(func, limits, n_dims, method="AUTO", dtype=tf.float64,
                   mc_sampler=tfp.mcmc.sample_halton_sequence,
                   mc_options=None):
    if method == "AUTO":  # TODO
        method = "mc"
    # TODO get n dims
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options['draws_per_dim']
        integral = mc_integrate(func=func, limits=limits, n_dims=n_dims, method=method, dtype=dtype,
                                mc_sampler=mc_sampler, draws_per_dim=draws_per_dim,
                                importance_sampling=None)
    return integral


# TODO
def numeric_integrate():
    """Integrate `func` using numerical methods."""
    integral = None
    return integral


def mc_integrate(func, limits, dims=None, value=None, n_dims=None, draws_per_dim=10000, method=None,
                 dtype=tf.float64,
                 mc_sampler=tfp.mcmc.sample_halton_sequence, importance_sampling=None):
    """

    Args:
        func ():
        limits ():
        dims (tuple(int)): The row to integrate over. None means integration over all values
        value ():
        n_dims ():
        draws_per_dim ():
        method ():
        dtype ():
        mc_sampler ():
        importance_sampling ():

    Returns:

    """
    lower, upper = limits  # TODO: limits?
    lower = tf.convert_to_tensor(lower, dtype=tf.float64)
    upper = tf.convert_to_tensor(upper, dtype=tf.float64)
    # HACK
    partial = (dims is not None) and (value is not None)  # dims, value can be tensors
    # if partial:
    #     raise ValueError("Partial integration not yet implemented.")
    if dims and n_dims:
        raise ValueError("Either specify dims or n_dims")
    if dims and not n_dims:
        n_dims = len(dims)
    n_samples = draws_per_dim ** n_dims
    if partial:
        n_vals = value.get_shape()[0].value
        n_samples *= n_vals  # each entry wants it's mc
    else:
        n_vals = 1
    print("DEBUG, n_dims", n_dims, dims, n_samples, dtype)
    # TODO: get dimensions properly
    # TODO: add times dim or so
    print("DEBUG, mc_sampler", mc_sampler)
    samples_normed = mc_sampler(dim=n_dims, num_results=n_samples, dtype=dtype)
    samples_normed = tf.reshape(samples_normed, shape=(n_vals, int(n_samples / n_vals), n_dims))
    print("DEBUG, samples_normed", samples_normed)
    samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
    samples = tf.reshape(samples, shape=(n_dims, int(n_samples / n_vals), n_vals))
    print("DEBUG, samples_normed", samples_normed)

    # TODO: combine sampled with values

    if partial:
        value_list = []
        index_samples = 0
        index_values = 0
        if len(value.shape) == 1:
            value = tf.expand_dims(value, axis=1)
        print("DEBUG, n_dims = ", n_dims, value.shape[1].value)
        for i in range(n_dims + value.shape[1].value):
            if i in dims:
                value_list.append(samples[:, index_samples])
                index_samples += 1
            else:
                value_list.append(value[:, index_values])
                index_values += 1
        value_list = [tf.cast(val, dtype=dtype) for val in value_list]
        # value = tf.stack(value_list)
        value = value_list
    else:
        value = samples

    # convert rnd samples with values to feedable vector
    reduce_axis = 1 if partial else None
    if partial:
        print("DEBUG, samples: ", samples)
        print("DEBUG, value: ", value)
        print("DEBUG, func(value): ", func(value))
        print("DEBUG, func(value).get_shape(): ", func(value).get_shape())
    avg = tf.reduce_mean(input_tensor=func(value), axis=reduce_axis)
    # avg = tfp.monte_carlo.expectation(f=func, samples=samples)
    print("DEBUG, avg", avg)
    integral = avg * tf.reduce_prod(upper - lower)
    return tf.cast(integral, dtype=dtype)


class AnalyticIntegral(object):
    def __init__(self, *args, **kwargs):
        super(AnalyticIntegral, self).__init__(*args, **kwargs)
        self._max_dims = []
        self._integrals = {}

    def get_max_dims(self, out_of_dims=None):
        if out_of_dims:
            out_of_dims = frozenset(out_of_dims)
            implemented_dims = set(d for d in self._integrals.keys() if d <= out_of_dims)
            max_dims = max(implemented_dims)
        else:
            max_dims = self._max_dims

        return max_dims

    def register(self, func, dims):
        """Register an analytic integral."""
        dims = frozenset(dims)
        if len(dims) > len(self.max_dims):
            self.max_dims = dims
        self._integrals[dims] = func

    def integrate(self, value, limits, dims):
        """Integrate analytically over the dims if available."""
        dims = dims or self.dims  # integrate over all dims
        dims = frozenset(dims)
        integral_fn = self._integrals.get(dims)
        if integral_fn is None:
            raise NotImplementedError("This integral is not available for dims {}".format(dims))
        integral = integral_fn(value=value, limits=limits)
        return integral


if __name__ == '__main__':
    # TODO: partial does not yet work...
    def my_fn1(value):
        if isinstance(value, tf.Tensor):
            value = tf.unstack(value)
        w, x, y, z, l = value
        # return x ** 2 + 0.1 * y ** 2 + 0.01 * z ** 2 + 0.001 * w ** 2 + 0.0001 * l ** 2
        return w + x


    import tensorflow_probability as tfp

    res = mc_integrate(func=my_fn1, limits=(0., 5.), dims=(1, 3), draws_per_dim=1000,
                       value=tf.constant([[i, i, i] for i in range(1, 6)]),
                       mc_sampler=tfp.mcmc.sample_halton_sequence)

    with tf.Session() as sess:
        res = sess.run(res)
        print(res)
