#  Copyright (c) 2019 zfit

from contextlib import suppress
from typing import Callable, Union, Iterable, List, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit import ztf
from zfit.core.interfaces import ZfitPDF
from zfit.util import ztyping
from zfit.util.exception import ShapeIncompatibleError, DueToLazynessNotImplementedError
from .. import settings
from ..util.container import convert_to_container
from .limits import Space
from ..settings import ztypes, run


class UniformSampleAndWeights:
    def __call__(self, n_to_produce: Union[int, tf.Tensor], limits: Space, dtype):
        rnd_samples = []
        thresholds_unscaled_list = []
        weights = tf.broadcast_to(ztf.constant(1., shape=(1,)), shape=(n_to_produce,))

        for (lower, upper), area in zip(limits.iter_limits(as_tuple=True), limits.iter_areas(rel=True)):
            n_partial_to_produce = tf.to_int32(
                ztf.to_real(n_to_produce) * ztf.to_real(area))  # TODO(Mayou36): split right!

            lower = ztf.convert_to_tensor(lower, dtype=dtype)
            upper = ztf.convert_to_tensor(upper, dtype=dtype)

            if isinstance(limits, EventSpace):
                lower = tf.transpose(lower)
                upper = tf.transpose(upper)

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


class EventSpace(Space):
    """EXPERIMENTAL SPACE CLASS!"""

    def __init__(self, obs: ztyping.ObsTypeInput, limits: ztyping.LimitsTypeInput, factory=None,
                 name: Optional[str] = "Space"):
        if limits is None:
            raise ValueError("Limits cannot be None for EventSpaces (currently)")
        self._limits_tensor = None
        self._factory = factory
        super().__init__(obs, limits, name)

    @property
    def factory(self):
        return self._factory

    @property
    def is_generator(self):
        return self.factory is not None

    @property
    def limits(self) -> ztyping.LimitsTypeReturn:
        limits = super().limits
        limits_tensor = self._limits_tensor
        if limits_tensor is not None:
            lower, upper = limits
            new_bounds = [[], []]
            for i, old_bounds in enumerate(lower, upper):
                for bound in old_bounds:
                    if self.is_generator:
                        new_bound = tuple(lim(limits_tensor) for lim in bound)
                    else:
                        new_bound = tuple(lim() for lim in bound)
                    new_bounds[i].append(new_bound)
                new_bounds[i] = tuple(new_bounds[i])
            limits = tuple(new_bounds)

        return limits

    def create_limits(self, n):
        if self._factory is not None:
            self._limits_tensor = self._factory(n)

    def iter_areas(self, rel: bool = False) -> Tuple[float, ...]:
        if not rel:
            raise RuntimeError("Currently, only rel with one limits is implemented in EventSpace")
        return (1.,)  # TODO: remove HACK, use tensors?
        raise RuntimeError("Cannot be called with an event space.")

    def add(self, other: ztyping.SpaceOrSpacesTypeInput):
        raise RuntimeError("Cannot be called with an event space.")

    def combine(self, other: ztyping.SpaceOrSpacesTypeInput):
        raise RuntimeError("Cannot be called with an event space.")

    @staticmethod
    def _calculate_areas(limits) -> Tuple[float]:
        # TODO: return the area as a tensor?
        return (1.,)


def accept_reject_sample(prob: Callable, n: int, limits: Space,
                         sample_and_weights_factory: Callable = UniformSampleAndWeights,
                         dtype=ztypes.float, prob_max: Union[None, int] = None,
                         efficiency_estimation: float = 1.0) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob (function): A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n (int): Number of samples to produce
        limits (:py:class:`~zfit.Space`): The limits to sample from
        sample_and_weights_factory (Callable): A factory function that returns the following function:
            A function that returns the sample to insert into `prob` and the weights
            (probability density) of each sample together with the random thresholds. The API looks as follows:

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

    sample_and_weights = sample_and_weights_factory()
    n = tf.to_int32(n)
    if run.numeric_checks:
        assert_valid_n_op = tf.assert_non_negative(n)
        deps = [assert_valid_n_op]
    else:
        deps = []
    # whether we may produce more then n, we normally do (except for EventSpace which is not a generator)
    # we cannot cut inside the while loop as soon as we have produced enough because we may sample from
    # multiple limits and therefore need to randomly remove events, otherwise we are biased because the
    # drawn samples are ordered in the different
    dynamic_array_shape = True

    # for fixed limits in EventSpace we need to know which indices have been successfully sampled. Therefore this
    # can be None (if not needed) or a boolean tensor with the size `n`
    initial_is_sampled = tf.constant("EMPTY")
    if isinstance(limits, EventSpace) and not limits.is_generator:
        dynamic_array_shape = False
        if run.numeric_checks:
            assert_n_matches_limits_op = tf.assert_equal(tf.shape(limits.lower[0][0])[0], n)
            tfdeps = [assert_n_matches_limits_op]
        else:
            tfdeps = []
        with tf.control_dependencies(tfdeps):  # TODO(Mayou36): good check? could be 1d
            initial_is_sampled = tf.fill(value=False, dims=(n,))
        efficiency_estimation = 1.0  # generate exactly n
    with tf.control_dependencies(deps):
        inital_n_produced = tf.constant(0, dtype=tf.int32)
        initial_n_drawn = tf.constant(0, dtype=tf.int32)
        with tf.control_dependencies([n]):
            sample = tf.TensorArray(dtype=dtype, size=n, dynamic_size=dynamic_array_shape,
                                    clear_after_read=True,  # we read only once at end to tensor
                                    element_shape=(limits.n_obs,))

    def not_enough_produced(n, sample, n_produced, n_total_drawn, eff, is_sampled, weights_scaling):
        return tf.greater(n, n_produced)

    def sample_body(n, sample, n_produced=0, n_total_drawn=0, eff=1.0, is_sampled=None, weights_scaling=0.):
        eff = tf.reduce_max([eff, ztf.to_real(1e-6)])

        n_to_produce = n - n_produced

        if isinstance(limits, EventSpace):  # EXPERIMENTAL(Mayou36): added to test EventSpace
            limits.create_limits(n=n)

        do_print = settings.get_verbosity() > 5
        if do_print:
            print_op = tf.print("Number of samples to produce:", n_to_produce, " with efficiency ", eff,
                                " with total produced ", n_produced, " and total drawn ", n_total_drawn,
                                " with weights scaling", weights_scaling)
        with tf.control_dependencies([print_op] if do_print else []):
            n_to_produce = tf.identity(n_to_produce)
        if dynamic_array_shape:
            n_to_produce = tf.to_int32(ztf.to_real(n_to_produce) / eff * (1.1)) + 10  # just to make sure
            # TODO: adjustable efficiency cap for memory efficiency (prevent too many samples at once produced)
            max_produce_cap = tf.to_int32(8e5)
            safe_to_produce = tf.maximum(max_produce_cap, n_to_produce)  # protect against overflow, n_to_prod -> neg.
            n_to_produce = tf.minimum(safe_to_produce, max_produce_cap)  # introduce a cap to force serial
            new_limits = limits
        else:
            # TODO(Mayou36): add cap for n_to_produce here as well
            if multiple_limits:
                raise DueToLazynessNotImplementedError("Multiple limits for fixed event space not yet implemented")
            is_not_sampled = tf.logical_not(is_sampled)
            (lower,), (upper,) = limits.limits
            lower = tuple(tf.boolean_mask(low, is_not_sampled) for low in lower)
            upper = tuple(tf.boolean_mask(up, is_not_sampled) for up in upper)
            new_limits = limits.with_limits(limits=((lower,), (upper,)))
            draw_indices = tf.where(is_not_sampled)

        with tf.control_dependencies([n_to_produce]):
            rnd_sample, thresholds_unscaled, weights, weights_max, n_drawn = sample_and_weights(
                n_to_produce=n_to_produce,
                limits=new_limits,
                dtype=dtype)

        n_drawn = tf.cast(n_drawn, dtype=tf.int32)
        if run.numeric_checks:
            assert_op_n_drawn = tf.assert_non_negative(n_drawn)
            tfdeps = [assert_op_n_drawn]
        else:
            tfdeps = []
        with tf.control_dependencies(tfdeps):
            n_total_drawn += n_drawn

            probabilities = prob(rnd_sample)
        shape_rnd_sample = tf.shape(rnd_sample)[0]
        if run.numeric_checks:
            assert_prob_rnd_sample_op = tf.assert_equal(tf.shape(probabilities), shape_rnd_sample)
            tfdeps = [assert_prob_rnd_sample_op]
        else:
            tfdeps = []
        # assert_weights_rnd_sample_op = tf.assert_equal(tf.shape(weights), shape_rnd_sample)
        # print_op = tf.print("shapes: ", tf.shape(weights), shape_rnd_sample, "shapes end")
        with tf.control_dependencies(tfdeps):
            probabilities = tf.identity(probabilities)
        if prob_max is None or weights_max is None:  # TODO(performance): estimate prob_max, after enough estimations -> fix it?
            # TODO(Mayou36): This control dependency is needed because otherwise the max won't be determined
            # correctly. A bug report on will be filled (WIP).
            # The behavior is very odd: if we do not force a kind of copy, the `reduce_max` returns
            # a value smaller by a factor of 1e-14
            # with tf.control_dependencies([probabilities]):
            # UPDATE: this works now? Was it just a one-time bug?

            # safety margin, predicting future, improve for small samples?
            weights_maximum = tf.reduce_max(weights)
            weights_clipped = tf.maximum(weights, weights_maximum * 1e-5)
            # prob_weights_ratio = probabilities / weights
            prob_weights_ratio = probabilities / weights_clipped
            # min_prob_weights_ratio = tf.reduce_min(prob_weights_ratio)
            max_prob_weights_ratio = tf.reduce_max(prob_weights_ratio)
            ratio_threshold = 50000000.
            # clipping means that we don't scale more for a certain threshold
            # to properly account for very small numbers, the thresholds should be scaled to match the ratio
            # but if a weight of a sample is very low (compared to the other weights), this would force the acceptance
            # of other samples to decrease strongly. We introduce a cut here, meaning that any event with an acceptance
            # chance of less then 1 in ratio_threshold will be underestimated.
            # TODO(Mayou36): make ratio_threshold a global setting
            # max_prob_weights_ratio_clipped = tf.minimum(max_prob_weights_ratio,
            #                                             min_prob_weights_ratio * ratio_threshold)
            max_prob_weights_ratio_clipped = max_prob_weights_ratio
            weights_scaling = tf.maximum(weights_scaling, max_prob_weights_ratio_clipped * (1 + 1e-2))
        else:
            weights_scaling = prob_max / weights_max
            min_prob_weights_ratio = weights_scaling

        weights_scaled = weights_scaling * weights * (1 + 1e-8)  # numerical epsilon
        random_thresholds = thresholds_unscaled * weights_scaled
        if run.numeric_checks:
            invalid_probs_weights = tf.greater(probabilities, weights_scaled)
            failed_weights = tf.boolean_mask(weights_scaled, mask=invalid_probs_weights)
            failed_probs = tf.boolean_mask(probabilities, mask=invalid_probs_weights)

            print_op = tf.print("HACK WARNING: if the following is NOT empty, your sampling _may_ be biased."
                                " Failed weights:", failed_weights, " failed probs", failed_probs)
            assert_no_failed_probs = tf.assert_equal(tf.shape(failed_weights), [0])
            # assert_op = [print_op]
            assert_op = [assert_no_failed_probs]
            # for weights scaled more then ratio_threshold
            # assert_op = [tf.assert_greater_equal(x=weights_scaled, y=probabilities,
            #                                      data=[tf.shape(failed_weights), failed_weights, failed_probs],
            #                                      message="Not all weights are >= probs so the sampling "
            #                                              "will be biased. If a custom `sample_and_weights` "
            #                                              "was used, make sure that either the shape of the "
            #                                              "custom sampler (resp. it's weights) overlap better "
            #                                              "or decrease the `max_weight`")]
            #
            # # check disabled (below not added to deps)
            # assert_scaling_op = tf.assert_less(weights_scaling / min_prob_weights_ratio, ztf.constant(ratio_threshold),
            #                                    data=[weights_scaling, min_prob_weights_ratio],
            #                                    message="The ratio between the probabilities from the pdf and the"
            #                                    f"probability from the sampler is higher "
            #                                    f" then {ratio_threshold}. This will most probably bias the sampling. "
            #                                    f"Use importance sampling or, to disable this check, do"
            #                                    f"zfit.run.numeric_checks = False")
            # assert_op.append(assert_scaling_op)
        else:
            assert_op = []
        with tf.control_dependencies(assert_op):
            take_or_not = probabilities > random_thresholds
        take_or_not = take_or_not[0] if len(take_or_not.shape) == 2 else take_or_not
        filtered_sample = tf.boolean_mask(rnd_sample, mask=take_or_not, axis=0)

        n_accepted = tf.shape(filtered_sample)[0]
        n_produced_new = n_produced + n_accepted
        if not dynamic_array_shape:
            indices = tf.boolean_mask(draw_indices, mask=take_or_not)
            current_sampled = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=indices,
                                                                        values=tf.broadcast_to(input=(True,),
                                                                                               shape=(n_accepted,)),
                                                                        dense_shape=(tf.cast(n, dtype=tf.int64),)),
                                                        default_value=False)
            is_sampled = tf.logical_or(is_sampled, current_sampled)
            indices = indices[:, 0]
        else:
            indices = tf.range(n_produced, n_produced_new)

        sample_new = sample.scatter(indices=tf.cast(indices, dtype=tf.int32), value=filtered_sample)

        # efficiency (estimate) of how many samples we get
        eff = tf.reduce_max([ztf.to_real(n_produced_new), ztf.to_real(1.)]) / tf.reduce_max(
            [ztf.to_real(n_total_drawn), ztf.to_real(1.)])
        return n, sample_new, n_produced_new, n_total_drawn, eff, is_sampled, weights_scaling

    efficiency_estimation = ztf.to_real(efficiency_estimation)
    weights_scaling = ztf.constant(0.)
    loop_vars = (
        n, sample, inital_n_produced, initial_n_drawn, efficiency_estimation, initial_is_sampled, weights_scaling)

    sample_array = tf.while_loop(cond=not_enough_produced, body=sample_body,  # paraopt
                                 loop_vars=loop_vars,
                                 swap_memory=True,
                                 parallel_iterations=1,
                                 back_prop=False)[1]  # backprop not needed here
    new_sample = sample_array.stack()
    if multiple_limits:
        new_sample = tf.random.shuffle(new_sample)  # to make sure, randomly remove and not biased.
    if dynamic_array_shape:  # if not dynamic we produced exact n -> no need to cut
        new_sample = new_sample[:n, :]  # cutting away to many produced

    # if no failure, uncomment both for improvement of shape inference, but what if n is tensor?
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
        limits (:py:class:`~zfit.Space`):

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
