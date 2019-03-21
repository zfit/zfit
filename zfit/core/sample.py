from contextlib import suppress
from typing import Callable, Union, Iterable, List

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit import ztf
from zfit.core.interfaces import ZfitPDF
from zfit.util.exception import ShapeIncompatibleError
from .. import settings
from ..util.container import convert_to_container
from .limits import Space
from ..settings import ztypes, run


def uniform_sample_and_weights(n_to_produce: Union[int, tf.Tensor], limits: Space, dtype):
    rnd_samples = []
    thresholds_unscaled_list = []
    weights = ztf.constant(1., shape=(1,))

    for (lower, upper), area in zip(limits.iter_limits(as_tuple=True), limits.iter_areas(rel=True)):
        n_partial_to_produce = tf.to_int64(ztf.to_real(n_to_produce) * ztf.to_real(area))
        lower = ztf.convert_to_tensor(lower, dtype=dtype)
        upper = ztf.convert_to_tensor(upper, dtype=dtype)
        sample_drawn = tf.random_uniform(shape=(n_partial_to_produce, limits.n_obs + 1),
                                         # + 1 dim for the function value
                                         dtype=ztypes.float)
        rnd_sample = sample_drawn[:, :-1] * (upper - lower) + lower  # -1: all except func value
        thresholds_unscaled = sample_drawn[:, -1]
        # if not multiple_limits:
        #     return rnd_sample, thresholds_unscaled
        rnd_samples.append(rnd_sample)
        thresholds_unscaled_list.append(thresholds_unscaled)

    rnd_sample = tf.concat(rnd_samples, axis=0)
    thresholds_unscaled = tf.concat(thresholds_unscaled_list, axis=0)

    n_drawn = n_to_produce
    return rnd_sample, thresholds_unscaled, weights, weights, n_drawn


def accept_reject_sample(prob: Callable, n: int, limits: Space,
                         sample_and_weights: Callable = uniform_sample_and_weights,
                         dtype=ztypes.float, prob_max: Union[None, int] = None,
                         efficiency_estimation: float = 1.0) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob (function): A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n (int): Number of samples to produce
        limits (Space): The limits to sample from
        sample_and_weights (Callable): A function that returns the sample to insert into `prob` and the weights
            (prob) of each sample together with the random thresholds. The API looks as follows:

            - Parameters:

                - n_to_produce (int, tf.Tensor): The number of events to produce (not exactly).
                - limits (Space): the limits in which the samples will be.
                - dtype (dtype): DType of the output.

            - Return:
                A tuple of length 5:
                - proposed sample (tf.Tensor with shape=(n_to_produce, n_obs)): The new (proposed) sample
                    whose values are inside `limits`.
                - thresholds_unscaled (tf.Tensor with shape=(n_to_produce,): Uniformly distributed
                    random values **between 0 and 1**.
                - weights (tf.Tensor with shape=(n_to_produce)): (Proportional to the) probability
                    for each sample of the distribution it was drawn from.
                - weights_max (int, tf.Tensor, None): The maximum of the weights (if known). This is
                    what the probability maximum will be scaled with, so it should be rather lower than the maximum
                    if the peaks do not exactly coincide. Otherwise return None (which will **assume**
                    that the peaks coincide).
                - n_produced: the number of events produced. Can deviate from the requested number.

        dtype ():
        prob_max (Union[None, int]): The maximum of the model function for the given limits. If None
            is given, it will be automatically, safely estimated (by a 10% increase in computation time
            (constant weak scaling)).
        efficiency_estimation (float): estimation of the initial sampling efficiency.

    Returns:
        tf.Tensor:
    """
    multiple_limits = limits.n_limits > 1

    # if limits.n_limits == 1:
    #     lower, upper = limits.limits
    #     lower = ztf.convert_to_tensor(lower[0], dtype=dtype)
    #     upper = ztf.convert_to_tensor(upper[0], dtype=dtype)

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

        rnd_sample, thresholds_unscaled, weights, weights_max, n_drawn = sample_and_weights(n_to_produce=n_to_produce,
                                                                                            limits=limits,
                                                                                            dtype=dtype)

        # if n_produced is None:
        #     raise ShapeIncompatibleError("`sample_and_weights` has to return thresholds with a defined shape."
        #                                  "Use `Tensor.set_shape()` if the automatic propagation of the shape "
        #                                  "is not available.")
        n_total_drawn += n_drawn
        n_total_drawn = tf.to_int64(n_total_drawn)

        probabilities = prob(rnd_sample)
        if prob_max is None:  # TODO(performance): estimate prob_max, after enough estimations -> fix it?
            # TODO(Mayou36): This control dependency is needed because otherwise the max won't be determined
            # correctly. A bug report on will be filled (WIP).
            # The behavior is very odd: if we do not force a kind of copy, the `reduce_max` returns
            # a value smaller by a factor of 1e-14
            # with tf.control_dependencies([probabilities]):
            # UPDATE: this works now? Was it just a one-time bug?
                prob_max_inferred = tf.reduce_max(probabilities)
        else:
            prob_max_inferred = prob_max

        if weights_max is None:
            weights_max = tf.reduce_max(weights) * 0.99  # safety margin, also taking numericals into account

        weights_scaled = prob_max_inferred / weights_max * weights
        random_thresholds = thresholds_unscaled * weights_scaled
        if run.numeric_checks:
            assert_op = [tf.assert_greater_equal(x=weights_scaled, y=probabilities,
                                                 message="Not all weights are >= probs so the sampling "
                                                         "will be biased. If a custom `sample_and_weights` "
                                                         "was used, make sure that either the shape of the "
                                                         "custom sampler (resp. it's weights) overlap better "
                                                         "or decrease the `max_weight`")]
        else:
            assert_op = []
        with tf.control_dependencies(assert_op):
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


def extract_extended_pdfs(pdfs: Union[Iterable[ZfitPDF], ZfitPDF]) -> List[ZfitPDF]:
    """Return all extended pdfs that are daughters.

    Args:
        pdfs (Iterable[pdfs]):

    Returns:
        List[pdfs]:
    """
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


def extended_sampling(pdfs: Union[Iterable[ZfitPDF], ZfitPDF], limits: Space) -> tf.Tensor:
    """Create a sample from extended pdfs by sampling poissonian using the yield.

    Args:
        pdfs (iterable[ZfitPDF]):
        limits (zfit.Space):

    Returns:
        Union[tf.Tensor]:
    """
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
