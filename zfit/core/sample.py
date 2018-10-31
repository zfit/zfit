from __future__ import print_function, division, absolute_import

import typing

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit import ztf
from zfit.core.limits import Range, no_multiple_limits
from ..settings import types as ztypes


@no_multiple_limits
def accept_reject_sample(prob: typing.Callable, n_draws: int, limits: Range,
                         sampler: typing.Callable = tf.random_uniform,
                         dtype=ztypes.float,
                         prob_max: typing.Union[None, int] = None) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob (function): A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n_draws (int): Number of samples to produce
        limits (Range): The limits to sample from
        sampler (function): A function taking n_draws as an argument and returning values between
            0 and 1
        dtype ():
        prob_max (Union[None, int]): The maximum of the prob function for the given limits. If None
            is given, it will be automatically, safely estimated (by a 10% increase in computation time
            (constant weak scaling)).

    Returns:
        tensorflow.python.framework.ops.Tensor:
    """
    n_dims = limits.n_dims
    lower, upper = limits.get_boundaries()
    lower = tf.convert_to_tensor(lower, dtype=dtype)
    upper = tf.convert_to_tensor(upper, dtype=dtype)
    n_draws = tf.to_int64(n_draws)

    def enough_produced(n_draws, sample, n_total_drawn, eff):
        return tf.less(tf.shape(sample, out_type=tf.int64)[1], n_draws)

    def sample_body(n_draws, sample, n_total_drawn=0, eff=1.):
        if sample is None:
            n_to_produce = n_draws
        else:
            n_to_produce = n_draws - tf.shape(sample, out_type=tf.int64)[1]
        # TODO: add efficiency cap for memory efficiency (prevent too many samples at once produced)
        n_to_produce = tf.to_int64(ztf.to_float(n_to_produce) / eff * 0.99) + 100  # just to make sure
        n_total_drawn += n_to_produce
        n_total_drawn = tf.to_int64(n_total_drawn)

        sample_drawn = sampler(shape=(n_dims + 1, n_to_produce),  # + 1 dim for the function value
                               dtype=ztypes.float)
        rnd_sample = sample_drawn[:-1, :] * (upper - lower) + lower

        probabilities = prob(rnd_sample)
        if prob_max is None:
            prob_max_inferred = tf.reduce_max(probabilities)
        else:
            prob_max_inferred = prob_max
        random_thresholds = sample_drawn[-1, :] * prob_max_inferred

        take_or_not = probabilities > random_thresholds

        filtered_sample = tf.boolean_mask(rnd_sample, mask=take_or_not[0], axis=1)

        if sample is None:
            sample = filtered_sample
        else:
            sample = tf.concat([sample, filtered_sample], axis=1)

        # efficiency (estimate) of how many samples we get
        eff = ztf.to_float(tf.shape(sample, out_type=tf.int64)[1]) / ztf.to_float(n_total_drawn)
        return n_draws, sample, n_total_drawn, eff


    sample = tf.while_loop(cond=enough_produced, body=sample_body,  # paraopt
                           loop_vars=sample_body(n_draws=n_draws, sample=None,  # run first once for initialization
                                                 n_total_drawn=0, eff=1.))[1]
    return sample[:, :n_draws]  # cutting away to many produced


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import time

    with tf.Session() as sess:
        dist = tfp.distributions.Normal(loc=1.5, scale=5.)
        log_prob_fn = dist.log_prob
        hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=log_prob_fn, step_size=0.1,
                                             num_leapfrog_steps=2)
        samples, kernel_results = tfp.mcmc.sample_chain(num_results=int(2),
                                                        num_burnin_steps=int(3),
                                                        num_steps_between_results=int(3),
                                                        current_state=0.3, kernel=hmc,
                                                        parallel_iterations=80)

        init = tf.global_variables_initializer()
        sess.run(init)
        result = sess.run(samples)
        print(np.average(result), np.std(result))
        # maximum = 1.1 * tf.reduce_max(dist.prob(tf.random_uniform((10000,), -100, 100)))
        maximum = 0.1
        # maximum = None
        list1 = []

        sampled_dist_ar = accept_reject_sample(prob=dist.prob, n_draws=100000000,
                                               limits=(-13.5, 16.5),
                                               prob_max=maximum, sampler=None)

        # HACK

        start = time.time()
        for _ in range(40):
            _ = sess.run(sampled_dist_ar)
        end = time.time()
        print("Time needed for normalization:", end - start)
        # plt.hist(sampled_dist_ar, bins=40)

        plt.figure()
        # plt.hist(result, bins=40)
        plt.show()
