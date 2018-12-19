import typing

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit import ztf
from zfit.core.limits import Space, no_multiple_limits
from ..settings import types as ztypes


@no_multiple_limits
def accept_reject_sample(prob: typing.Callable, n: int, limits: Space,
                         sampler: typing.Callable = tf.random_uniform,
                         dtype=ztypes.float, prob_max: typing.Union[None, int] = None) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob (function): A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n (int): Number of samples to produce
        limits (Space): The limits to sample from
        sampler (function): A function taking n as an argument and returning value between
            0 and 1
        dtype ():
        prob_max (Union[None, int]): The maximum of the model function for the given limits. If None
            is given, it will be automatically, safely estimated (by a 10% increase in computation time
            (constant weak scaling)).

    Returns:
        tf.Tensor:
    """
    n_dims = limits.n_obs
    lower, upper = limits.limits
    lower = ztf.convert_to_tensor(lower[0], dtype=dtype)
    upper = ztf.convert_to_tensor(upper[0], dtype=dtype)
    n = tf.to_int64(n)

    def enough_produced(n, sample, n_total_drawn, eff):
        return tf.less(tf.shape(sample, out_type=tf.int64)[1], n)

    def sample_body(n, sample, n_total_drawn=0, eff=1.):
        if sample is None:
            n_to_produce = n
        else:
            n_to_produce = n - tf.shape(sample, out_type=tf.int64)[1]
        # TODO: add efficiency cap for memory efficiency (prevent too many samples at once produced)
        n_to_produce = tf.to_int64(ztf.to_real(n_to_produce) / eff * 0.99) + 100  # just to make sure
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
        eff = ztf.to_real(tf.shape(sample, out_type=tf.int64)[1]) / ztf.to_real(n_total_drawn)
        return n, sample, n_total_drawn, eff

    sample = tf.while_loop(cond=enough_produced, body=sample_body,  # paraopt
                           loop_vars=sample_body(n=n, sample=None,  # run first once for initialization
                                                 n_total_drawn=0, eff=1.),
                           swap_memory=True, parallel_iterations=4,
                           back_prop=False)[1]  # backprop not needed here
    return sample[:, :n]  # cutting away to many produced


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
        # maximum = 1.1 * tf.reduce_max(dist.model(tf.random_uniform((10000,), -100, 100)))
        maximum = 0.1
        # maximum = None
        list1 = []

        sampled_dist_ar = accept_reject_sample(prob=dist.prob, n=100000000, limits=(-13.5, 16.5), sampler=None,
                                               prob_max=maximum)

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
