from contextlib import suppress
import typing

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit import ztf
from .. import settings
from ..util.container import convert_to_container
from .limits import Space
from ..settings import ztypes


def accept_reject_sample(prob: typing.Callable, n: int, limits: Space,
                         sampler: typing.Callable = tf.random_uniform,
                         dtype=ztypes.float, prob_max: typing.Union[None, int] = None,
                         efficiency_estimation: float = 1.0) -> tf.Tensor:
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
        efficiency_estimation (float): estimation of the initial sampling efficiency.

    Returns:
        tf.Tensor:
    """
    n_dims = limits.n_obs
    multiple_limits = limits.n_limits > 1

    # if limits.n_limits == 1:
    #     lower, upper = limits.limits
    #     lower = ztf.convert_to_tensor(lower[0], dtype=dtype)
    #     upper = ztf.convert_to_tensor(upper[0], dtype=dtype)

    def random_sampling(n_to_produce):
        rnd_samples = []
        thresholds_unscaled_list = []
        for (lower, upper), area in zip(limits.iter_limits(as_tuple=True), limits.iter_areas(rel=True)):
            n_partial_to_produce = tf.to_int64(ztf.to_real(n_to_produce) * ztf.to_real(area))
            lower = ztf.convert_to_tensor(lower[0], dtype=dtype)
            upper = ztf.convert_to_tensor(upper[0], dtype=dtype)
            sample_drawn = sampler(shape=(n_partial_to_produce, n_dims + 1),  # + 1 dim for the function value
                                   dtype=ztypes.float)
            rnd_sample = sample_drawn[:, :-1] * (upper - lower) + lower  # -1: all except func value
            thresholds_unscaled = sample_drawn[:, -1]
            if not multiple_limits:
                return rnd_sample, thresholds_unscaled
            rnd_samples.append(rnd_sample)
            thresholds_unscaled_list.append(thresholds_unscaled)

        rnd_sample = tf.concat(rnd_samples, axis=0)
        thresholds_unscaled = tf.concat(thresholds_unscaled_list, axis=0)
        return rnd_sample, thresholds_unscaled

    n = tf.to_int64(n)

    def enough_produced(n, sample, n_total_drawn, eff):
        return tf.greater(n, tf.shape(sample, out_type=tf.int64)[0])

    def sample_body(n, sample, n_total_drawn=0, eff=1.0):
        if sample is None:
            n_to_produce = n
        else:
            n_to_produce = n - tf.shape(sample, out_type=tf.int64)[0]
        do_print = settings.get_verbosity() > 5
        if do_print:
            print_op = tf.print("Number of samples to produce:", n_to_produce, " with efficiency ", eff)
        with tf.control_dependencies([print_op] if do_print else []):
            n_to_produce = tf.to_int64(ztf.to_real(n_to_produce) / eff * 1.01) + 100  # just to make sure
        # TODO: adjustable efficiency cap for memory efficiency (prevent too many samples at once produced)
        n_to_produce = tf.minimum(n_to_produce, tf.to_int64(5e5))  # introduce a cap to force serial
        n_total_drawn += n_to_produce
        n_total_drawn = tf.to_int64(n_total_drawn)

        rnd_sample, thresholds_unscaled = random_sampling(n_to_produce)
        probabilities = prob(rnd_sample)
        if prob_max is None:  # TODO(performance): estimate prob_max, after enough estimations -> fix it?
            prob_max_inferred = tf.reduce_max(probabilities)
        else:
            prob_max_inferred = prob_max
        random_thresholds = thresholds_unscaled * prob_max_inferred
        take_or_not = probabilities > random_thresholds
        # rnd_sample = tf.expand_dims(rnd_sample, dim=0) if len(rnd_sample.shape) == 1 else rnd_sample
        take_or_not = take_or_not[0] if len(take_or_not.shape) == 2 else take_or_not
        filtered_sample = tf.boolean_mask(rnd_sample, mask=take_or_not, axis=0)

        if sample is None:
            sample = filtered_sample
        else:
            sample = tf.concat([sample, filtered_sample], axis=0)

        # efficiency (estimate) of how many samples we get
        eff = ztf.to_real(tf.shape(sample, out_type=tf.int64)[1]) / ztf.to_real(n_total_drawn)
        return n, sample, n_total_drawn, eff

    sample = tf.while_loop(cond=enough_produced, body=sample_body,  # paraopt
                           loop_vars=sample_body(n=n, sample=None,  # run first once for initialization
                                                 n_total_drawn=0, eff=efficiency_estimation),
                           swap_memory=True,
                           parallel_iterations=4,
                           back_prop=False)[1]  # backprop not needed here
    if multiple_limits:
        sample = tf.random.shuffle(sample)  # to make sure, randomly remove and not biased.
    new_sample = sample[:n, :]  # cutting away to many produced

    # TODO(Mayou36): uncomment below. Why was set_shape needed? leave away to catch failure over time
    # if no failure, uncomment both for improvement of shape inference
    # with suppress(AttributeError):  # if n_samples_int is not a numpy object
    #     new_sample.set_shape((n_samples_int, n_dims))
    return new_sample


def extract_extended_pdfs(pdfs):
    from ..models.functor import BaseFunctor

    pdfs = convert_to_container(pdfs)
    indep_pdfs = []

    for pdf in pdfs:
        if not pdf.is_extended:
            continue
        elif isinstance(pdf, BaseFunctor):
            if all(pdf.pdfs_extended):
                indep_pdfs.extend(extract_extended_pdfs(pdfs=pdf.pdfs))
            elif not any(pdf.pdfs_extended):
                indep_pdfs.append(pdf)
            else:
                assert False, "Should not reach this point, wrong assumptions. Please report bug."
        else:  # extended, but not a functor
            indep_pdfs.append(pdf)

    return indep_pdfs


def extended_sampling(pdfs, limits):
    samples = []
    pdfs = convert_to_container(pdfs)
    pdfs = extract_extended_pdfs(pdfs)

    for pdf in pdfs:
        n = tf.random.poisson(lam=pdf.get_yield(), shape=(), dtype=ztypes.float)
        sample = pdf._single_hook_sample(limits=limits, n=n, name="extended_sampling")
        # sample.set_shape((n, limits.n_obs))
        samples.append(sample)

    samples = tf.concat(samples, axis=0)
    return samples


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

        result = sess.run(samples)
        print(np.average(result), np.std(result))
        # maximum = 1.1 * tf.reduce_max(dist.model(tf.random_uniform((10000,), -100, 100)))
        maximum = 0.1
        # maximum = None
        list1 = []

        sampled_dist_ar = accept_reject_sample(prob=dist.prob, n=100000000, limits=(-13.5, 16.5), sampler=None,
                                               prob_max=maximum)

        start = time.time()
        for _ in range(40):
            _ = sess.run(sampled_dist_ar)
        end = time.time()
        print("Time needed for normalization:", end - start)
        # plt.hist(sampled_dist_ar, bins=40)

        plt.figure()
        # plt.hist(result, bins=40)
        plt.show()
