from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import matplotlib.pyplot as plt

dist = tfp.distributions.Normal(loc=1.5, scale=5.)
log_prob_fn = dist.log_prob
hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=log_prob_fn, step_size=1.,
                                     num_leapfrog_steps=3)
samples, kernel_results = tfp.mcmc.sample_chain(num_results=int(10e3), num_burnin_steps=int(10e2),
                                                current_state=0.2, kernel=hmc,
                                                parallel_iterations=100)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(samples)
    print(np.average(result), np.std(result))

    plt.hist(result, bins=40)
    plt.show()
