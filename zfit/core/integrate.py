from __future__ import print_function, division, absolute_import

import collections

import tensorflow as tf
import tensorflow_probability as tfp

from .limits import convert_to_range, Range, no_norm_range
from ..settings import types as ztypes


@no_norm_range
def auto_integrate(func, limits, n_dims, x=None, method="AUTO", dtype=tf.float64,
                   mc_sampler=tfp.mcmc.sample_halton_sequence,
                   mc_options=None):
    if method == "AUTO":  # TODO
        method = "mc"
    # TODO get n dims
    # TODO method
    if method.lower() == "mc":
        mc_options = mc_options or {}
        draws_per_dim = mc_options['draws_per_dim']
        integral = mc_integrate(x=x, func=func, limits=limits, n_dims=n_dims, method=method, dtype=dtype,
                                mc_sampler=mc_sampler, draws_per_dim=draws_per_dim,
                                importance_sampling=None)
    return integral


# TODO
def numeric_integrate():
    """Integrate `func` using numerical methods."""
    integral = None
    return integral


def mc_integrate(func, limits, dims=None, x=None, n_dims=None, draws_per_dim=10000, method=None,
                 dtype=ztypes.float,
                 mc_sampler=tfp.mcmc.sample_halton_sequence, importance_sampling=None):
    """

    Args:
        func ():
        limits ():
        dims (tuple(int)): The row to integrate over. None means integration over all values
        x ():
        n_dims ():
        draws_per_dim ():
        method ():
        dtype ():
        mc_sampler ():
        importance_sampling ():

    Returns:

    """
    if dims is not None and n_dims is not None:
        raise ValueError("Either specify dims or n_dims")
    limits = convert_to_range(limits, dims)
    dims = limits.dims
    # HACK
    partial = (dims is not None) and (x is not None)  # dims, value can be tensors
    # if partial:
    #     raise ValueError("Partial integration not yet implemented.")
    if dims is not None and n_dims is None:
        n_dims = len(dims)
    if n_dims is not None and dims is None:
        dims = tuple(range(n_dims))

    lower, upper = limits.get_boundaries()
    # print("DEBUG": lower, upper", lower, upper)

    lower = tf.convert_to_tensor(lower, dtype=dtype)
    upper = tf.convert_to_tensor(upper, dtype=dtype)
    # print("DEBUG": lower, upper", lower, upper)

    n_samples = draws_per_dim ** n_dims
    if partial:
        n_vals = x.get_shape()[0].value
        n_samples *= n_vals  # each entry wants it's mc
    else:
        n_vals = 1
    # print("DEBUG", n_dims", n_dims, dims, n_samples, dtype)
    # TODO: get dimensions properly
    # TODO: add times dim or so
    # print("DEBUG", mc_sampler", mc_sampler)
    samples_normed = mc_sampler(dim=n_dims, num_results=n_samples, dtype=dtype)
    samples_normed = tf.reshape(samples_normed, shape=(n_vals, int(n_samples / n_vals), n_dims))
    # print("DEBUG", samples_normed", samples_normed)
    samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
    samples = tf.transpose(samples, perm=[2, 0, 1])
    # print("DEBUG", samples_normed", samples)

    # TODO: combine sampled with values

    if partial:
        value_list = []
        index_samples = 0
        index_values = 0
        if len(x.shape) == 1:
            # print("DEBUG", expanding dims n1")
            x = tf.expand_dims(x, axis=1)
        # print("DEBUG", n_dims = ", n_dims, x.shape[1].value)
        for i in range(n_dims + x.shape[1].value):
            if i in dims:
                value_list.append(samples[index_samples, :, :])
                index_samples += 1
            else:
                value_list.append(tf.expand_dims(x[:, index_values], axis=1))
                index_values += 1
        value_list = [tf.cast(val, dtype=dtype) for val in value_list]
        # value = tf.stack(value_list)
        x = value_list
    else:
        x = samples

    # convert rnd samples with values to feedable vector
    reduce_axis = 1 if partial else None
    if partial:
        pass
        # print("DEBUG", samples: ", samples)
        # print("DEBUG", value: ", x)
        # print("DEBUG", func(value): ", func(x))
        # print("DEBUG", func(value).get_shape(): ", func(x).get_shape())
    # avg = tf.reduce_mean(input_tensor=func(value), axis=reduce_axis)

    avg = tfp.monte_carlo.expectation(f=func, samples=x, axis=reduce_axis)
    # avg = tfb.monte_carlo.expectation_importance_sampler(f=func, samples=value,axis=reduce_axis)
    # print("DEBUG", avg", avg)
    integral = avg * limits.area
    return tf.cast(integral, dtype=dtype)


class AnalyticIntegral(object):
    def __init__(self, *args, **kwargs):
        super(AnalyticIntegral, self).__init__(*args, **kwargs)
        self._max_dims = ()
        self._integrals = collections.defaultdict(dict)

    def get_max_dims(self, out_of_dims=None, limits=None):

        if out_of_dims:
            out_of_dims = frozenset(out_of_dims)
            implemented_dims = set(d for d in self._integrals.keys() if d <= out_of_dims)
            implemented_dims_limits = []
            for dims in implemented_dims:
                may_integral_fn = self._integrals[dims].get(limits, self._integrals[dims].get(None))
                if may_integral_fn is not None:
                    implemented_dims_limits.append(dims)

            max_dims = max(implemented_dims_limits)
        else:
            max_dims = self._max_dims

        return max_dims

    def register(self, func, dims, limits=None):
        """Register an analytic integral."""
        if limits is not None:
            limits = convert_to_range(limits, dims=dims)
            limits = limits.as_tuple()
        dims = frozenset(dims)
        if len(dims) > len(self._max_dims):
            self._max_dims = dims
        self._integrals[dims][limits] = func  # TODO improve with database-like access

    def integrate(self, x, limits, dims=None, norm_range=None, params=None):
        """Integrate analytically over the dims if available."""
        # TODO: what if dims is None?
        if dims is None:
            dims = limits.dims
        dims = frozenset(dims)
        integral_holder = self._integrals.get(dims)
        # print("DEBUG", self._integrals, dims", self._integrals, dims)
        if integral_holder is None:
            raise NotImplementedError("Integral is not available for dims {}".format(dims))
        integral_fn = integral_holder.get(limits.as_tuple(), integral_holder.get(None))
        if integral_fn is None:
            raise NotImplementedError(
                "Integral is available for dims {}, but not for limits {}".format(dims, limits))

        if x is None:
            # print("DEBUG": limits", limits)
            integral = integral_fn(limits=limits, params=params)
        else:
            integral = integral_fn(x=x, limits=limits, params=params)
        return integral


class Integral(object):
    def __init__(self, func, limits, dims):
        self.func = func
        self._set_dims(dims)


if __name__ == '__main__':
    # TODO: partial does not yet work...
    def my_fn1(x):
        if isinstance(x, tf.Tensor):
            x = tf.unstack(x)
        w, x, y, z, l = x
        # return x ** 2 + 0.1 * y ** 2 + 0.01 * z ** 2 + 0.001 * w ** 2 + 0.0001 * l ** 2
        return w + x


    import tensorflow_probability as tfp

    res = mc_integrate(func=my_fn1, limits=(0., 5.), dims=(1, 3), draws_per_dim=1000,
                       x=tf.constant([[i, i, i] for i in range(1, 6)]),
                       mc_sampler=tfp.mcmc.sample_halton_sequence)

    with tf.Session() as sess:
        res = sess.run(res)
        print(res)
