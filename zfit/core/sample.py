from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from ..settings import types as ztypes


def accept_reject_sample(prob, n_draws, limits, sampler=tf.random_uniform, dtype=ztypes.float,
                         prob_max=None):
    """Return toy MC sample graph using accept-reject method

    Args:
        prob (method):

    """
    n_dims = 1  # HACK
    lower, upper = limits.get_boundaries()
    lower = tf.convert_to_tensor(lower, dtype=dtype)
    upper = tf.convert_to_tensor(upper, dtype=dtype)

    sample = sampler(shape=(n_dims + 1, n_draws),  # + 1 dim for the function value
                     dtype=ztypes.float)
    rnd_sample = sample[:-1, :] * (upper - lower) + lower
    probabilities = prob(rnd_sample)
    if prob_max is None:
        prob_max = tf.reduce_max(probabilities)
    random_thresholds = sample[-1, :] * prob_max
    take_or_not = probabilities > random_thresholds
    filtered_sample = tf.boolean_mask(rnd_sample, take_or_not)
    return filtered_sample


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
