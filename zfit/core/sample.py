#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress

import tensorflow as tf
from tensorflow_probability import distributions as tfd

import zfit.z.numpy as znp

from .. import settings, z
from ..settings import run, ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import WorkInProgressError
from .data import Data
from .interfaces import ZfitPDF, ZfitSpace
from .space import Space


class UniformSampleAndWeights:
    def __call__(self, n_to_produce: int | tf.Tensor, limits: Space, dtype, prng=None):
        if dtype is not None and dtype != ztypes.float:
            msg = "Only float is supported for now."
            raise ValueError(msg)

        if prng is None:
            prng = z.random.get_prng()
        rnd_samples = []
        thresholds_unscaled_list = []
        weights = tf.broadcast_to(z.constant(1.0, shape=(1,)), shape=(n_to_produce,))
        n_produced = tf.constant(0, tf.int64)
        for i, space in enumerate(limits):
            lower, upper = space.v0.limits  # TODO: remove new space
            if i == len(limits) - 1:
                n_partial_to_produce = n_to_produce - n_produced  # to prevent roundoff errors, shortcut for 1 space
            else:
                if isinstance(space, EventSpace):
                    frac = 1.0  # TODO(Mayou36): remove hack for Eventspace
                else:
                    tot_area = limits.volume
                    frac = (space.volume / tot_area)[0]
                n_partial_to_produce = znp.asarray(
                    z.to_real(n_to_produce) * z.to_real(frac), dtype=tf.int64
                )  # TODO(Mayou36): split right!

            sample_drawn = prng.uniform(
                shape=(n_partial_to_produce, limits.n_obs + 1),
                # + 1 dim for the function value
                dtype=ztypes.float,
            )

            rnd_sample = sample_drawn[:, :-1] * (upper - lower) + lower  # -1: all except func value
            thresholds_unscaled = sample_drawn[:, -1]

            rnd_samples.append(rnd_sample)
            thresholds_unscaled_list.append(thresholds_unscaled)
            n_produced += n_partial_to_produce

        rnd_sample = znp.concatenate(rnd_samples, axis=0)
        thresholds_unscaled = znp.concatenate(thresholds_unscaled_list, axis=0)

        n_drawn = n_to_produce
        return rnd_sample, thresholds_unscaled, weights, weights, n_drawn


class EventSpace(Space):
    """EXPERIMENTAL SPACE CLASS!"""

    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        limits: ztyping.LimitsTypeInput,
        factory=None,
        dtype=ztypes.float,
        name: str | None = "Space",
    ):
        if limits is None:
            msg = "Limits cannot be None for EventSpaces (currently)"
            raise ValueError(msg)
        self._limits_tensor = None
        self.dtype = dtype
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
        if (limits_tensor := self._limits_tensor) is not None:
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

    def iter_areas(self, rel: bool = False) -> tuple[float, ...]:
        if not rel:
            msg = "Currently, only rel with one limits is implemented in EventSpace"
            raise RuntimeError(msg)
        return (1.0,)  # TODO: remove HACK, use tensors?

    def add(self, other: ztyping.SpaceOrSpacesTypeInput):  # noqa: ARG002
        msg = "Cannot be called with an event space."
        raise RuntimeError(msg)

    def combine(self, other: ztyping.SpaceOrSpacesTypeInput):  # noqa: ARG002
        msg = "Cannot be called with an event space."
        raise RuntimeError(msg)

    @staticmethod
    def _calculate_areas(limits) -> tuple[float]:  # noqa: ARG004
        # TODO: return the area as a tensor?
        return (1.0,)

    def __hash__(self):
        return id(self)


# TODO: estimate maximum initially?
@z.function(wraps="sample")
def accept_reject_sample(
    prob: Callable,
    n: int,
    limits: ZfitSpace,
    sample_and_weights_factory: Callable = UniformSampleAndWeights,
    dtype=ztypes.float,
    prob_max: None | int = None,
    efficiency_estimation: float = 0.5,
) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob: A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n: Number of samples to produce
        limits: The limits to sample from
        sample_and_weights_factory: An (immutable!) factory function that returns the following function:
            A function that returns the sample to insert into ``prob`` and the weights
            (probability density) of each sample together with the random thresholds. The API looks as follows:

            - Parameters:

                - n_to_produce (int, tf.Tensor): The number of events to produce (not exactly).
                - limits (Space): the limits in which the samples will be.
                - dtype (dtype): DType of the output.

            - Return:
                A tuple of length 5:
                - proposed sample (tf.Tensor with shape=(n_to_produce, n_obs)): The new (proposed) sample
                    whose values are inside ``limits``.
                - thresholds_unscaled (tf.Tensor with shape=(n_to_produce,): Uniformly distributed
                    random values **between 0 and 1**.
                - weights (tf.Tensor with shape=(n_to_produce)): (Proportional to the) probability
                    for each sample of the distribution it was drawn from.
                - weights_max (int, tf.Tensor, None): The maximum of the weights (if known). This is
                    what the probability maximum will be scaled with, so it should be rather lower than the maximum
                    if the peaks do not exactly coincide. Otherwise return None (which will **assume**
                    that the peaks coincide).
                - n_produced: the number of events produced. Can deviate from the requested number.

        dtype:
        prob_max: The maximum of the model function for the given limits. If None
            is given, it will be automatically, safely estimated (by a 10% increase in computation time
            (constant weak scaling)).
        efficiency_estimation: estimation of the initial sampling efficiency.

    Returns:
    """
    prob_max_init = prob_max
    multiple_limits = len(limits) > 1
    if prob_max is None:
        n_min_to_produce = 3000
        overestimate_factor_scaling = 1.25
    else:  # if an exact estimation is given
        n_min_to_produce = 0
        overestimate_factor_scaling = 1.001

    sample_and_weights = sample_and_weights_factory()
    n = znp.asarray(n, dtype=tf.int64)
    if run.numeric_checks:
        tf.debugging.assert_non_negative(n)

    # whether we may produce more then n, we normally do (except for EventSpace which is not a generator)
    # we cannot cut inside the while loop as soon as we have produced enough because we may sample from
    # multiple limits and therefore need to randomly remove events, otherwise we are biased because the
    # drawn samples are ordered in the different
    dynamic_array_shape = True

    # for fixed limits in EventSpace we need to know which indices have been successfully sampled. Therefore this
    # can be None (if not needed) or a boolean tensor with the size `n`
    initial_is_sampled = tf.constant("EMPTY")
    if (isinstance(limits, EventSpace) and not limits.is_generator) or limits.n_events > 1:
        dynamic_array_shape = False
        if run.numeric_checks:
            tf.debugging.assert_equal(znp.asarray(limits.n_events, dtype=tf.int64), n)

        initial_is_sampled = tf.fill(value=False, dims=(n,))
        efficiency_estimation = 1.0  # generate exactly n

    inital_n_produced = tf.constant(0, dtype=tf.int64)
    initial_n_drawn = tf.constant(0, dtype=tf.int64)
    sample = tf.TensorArray(
        dtype=dtype,
        size=znp.asarray(n, dtype=tf.int32),
        dynamic_size=dynamic_array_shape,
        clear_after_read=True,  # we read only once at end to tensor
        element_shape=(limits.n_obs,),
    )

    @z.function(wraps="tensor", keepalive=True)
    def not_enough_produced(
        n,
        sample,
        n_produced,
        n_total_drawn,
        eff,
        is_sampled,
        weights_scaling,
        weights_maximum,
        prob_maximum,
        n_min_to_produce,
    ):
        del sample, n_total_drawn, eff, is_sampled, weights_scaling, weights_maximum, prob_maximum, n_min_to_produce
        return tf.greater(n, n_produced)

    @z.function(wraps="tensor", keepalive=True)
    def sample_body(
        n,
        sample,
        n_produced=0,
        n_total_drawn=0,
        eff=1.0,
        is_sampled=None,
        weights_scaling=0.0,
        weights_maximum=0.0,
        prob_maximum=0.0,
        n_min_to_produce=10000,
    ):
        eff = znp.max([eff, z.to_real(1e-6)])
        n_to_produce = n - n_produced

        if isinstance(limits, EventSpace):  # EXPERIMENTAL(Mayou36): added to test EventSpace
            limits.create_limits(n=n)

        do_print = settings.get_verbosity() > 5
        if do_print:
            tf.print(
                "Number of samples to produce:",
                n_to_produce,
                " with efficiency ",
                eff,
                " with total produced ",
                n_produced,
                " and total drawn ",
                n_total_drawn,
                " with weights scaling",
                weights_scaling,
            )

        if dynamic_array_shape:
            # TODO: move all this fixed numbers out into settings
            # this section is non-trivial due to numerical issues with large numbers in n_to_produce
            # and small numbers in the efficiency. To prevent an overflow, we scale the efficiency up
            # and convert it to an int, effectively precision after the floating point.
            # Then we scale it down again.
            eff_precision: int = 100
            one_over_eff_int = znp.asarray(1.0 / eff * 1.01 * eff_precision, dtype=tf.int64)
            n_to_produce *= one_over_eff_int
            n_to_produce = znp.floor_divide(n_to_produce, eff_precision)
            # tf.debugging.assert_positive(n_to_produce_float, "n_to_produce went negative, overflow?")
            # n_to_produce = znp.asarray(n_to_produce_float, dtype=tf.int64) + 3  # just to make sure
            n_to_produce = znp.maximum(n_to_produce, n_min_to_produce)
            # TODO: adjustable efficiency cap for memory efficiency (prevent too many samples at once produced)
            max_produce_cap = tf.constant(800000, dtype=tf.int64)
            tf.debugging.assert_positive(n_to_produce, "n_to_produce went negative, overflow?")
            # TODO: remove below? was there due to overflow in tf?
            # n_to_produce = znp.maximum(5, n_to_produce)  # protect against overflow, n_to_prod -> neg.
            n_to_produce = znp.minimum(n_to_produce, max_produce_cap)  # introduce a cap to force serial
            new_limits = limits  # because limits in the vector space case can change
        else:
            # TODO(Mayou36): add cap for n_to_produce here as well
            if multiple_limits:
                msg = "Multiple limits for fixed event space not yet implemented"
                raise WorkInProgressError(msg)
            is_not_sampled = tf.logical_not(is_sampled)
            lower, upper = limits._rect_limits_tf
            lower = tf.boolean_mask(tensor=lower, mask=is_not_sampled)
            upper = tf.boolean_mask(tensor=upper, mask=is_not_sampled)
            new_limits = limits.with_limits(limits=(lower, upper))
            draw_indices = tf.where(is_not_sampled)

        (
            rnd_sample,
            thresholds_unscaled,
            weights,
            weights_max,
            n_drawn,
        ) = sample_and_weights(n_to_produce=n_to_produce, limits=new_limits, dtype=dtype)

        rnd_sample_data = Data.from_tensor(obs=new_limits, tensor=rnd_sample)
        n_drawn = znp.asarray(n_drawn, dtype=tf.int64)
        if run.numeric_checks:
            tf.debugging.assert_non_negative(n_drawn)
        n_total_drawn += n_drawn
        probabilities = prob(rnd_sample_data)
        shape_rnd_sample = tf.shape(input=rnd_sample)[0]
        if run.numeric_checks:
            tf.debugging.assert_equal(tf.shape(input=probabilities), shape_rnd_sample)

        if (
            prob_max_init is None or weights_max is None
        ):  # TODO(performance): estimate prob_max, after enough estimations -> fix it?
            # safety margin, predicting future, improve for small samples?
            weights_maximum_new = znp.max(weights)
            weights_maximum = znp.maximum(weights_maximum, weights_maximum_new)
            # if run.numeric_checks:
            #     tf.debugging.assert_greater_equal(weights, weights_maximum * 1e-5, message="The ")
            weights_clipped = znp.maximum(weights, weights_maximum * 1e-5)
            # prob_weights_ratio = probabilities / weights
            prob_maximum_new = znp.max(probabilities)
            prob_maximum = znp.maximum(prob_maximum, prob_maximum_new)
            # prob_weights_ratio = probabilities / weights_clipped
            prob_weights_ratio_max = znp.max(probabilities / weights_clipped)
            max_prob_weights_ratio = znp.maximum(prob_maximum / weights_maximum, prob_weights_ratio_max)
            # clipping means that we don't scale more for a certain threshold
            # to properly account for very small numbers, the thresholds should be scaled to match the ratio
            # but if a weight of a sample is very low (compared to the other weights), this would force the acceptance
            # of other samples to decrease strongly. We introduce a cut here, meaning that any event with an acceptance
            # chance of less then 1 in ratio_threshold will be underestimated.
            # TODO(Mayou36): make ratio_threshold a global setting
            # max_prob_weights_ratio_clipped = znp.minimum(max_prob_weights_ratio,
            #                                             min_prob_weights_ratio * ratio_threshold)
            # max_prob_weights_ratio_clipped = max_prob_weights_ratio
            new_scaling_needed = max_prob_weights_ratio > weights_scaling
            old_scaling = weights_scaling
            weights_scaling = tf.cond(
                new_scaling_needed,
                lambda: max_prob_weights_ratio * overestimate_factor_scaling,
                lambda: weights_scaling,
            )

            def calc_new_n_produced():
                n_produced_float = znp.asarray(n_produced, dtype=ztypes.float)
                binomial = tfd.Binomial(n_produced_float, probs=old_scaling / weights_scaling)
                return znp.asarray(tf.round(binomial.sample()), dtype=tf.int64)

            n_produced = tf.cond(new_scaling_needed, calc_new_n_produced, lambda: n_produced)
            # TODO: for fixed array shape?
            # weights_scaling = znp.maximum(max_prob_weights_ratio, weights_scaling)

            # weights_scaling = znp.maximum(weights_scaling, max_prob_weights_ratio_clipped * (1 + 1e-2))
        else:
            weights_scaling = prob_max / weights_max

        weights_scaled = (
            weights_scaling * weights
            # * (1 + 1e-8)
        )  # numerical epsilon
        random_thresholds = thresholds_unscaled * weights_scaled
        if run.numeric_checks:
            invalid_probs_weights = tf.greater(probabilities, weights_scaled)
            failed_weights = tf.boolean_mask(tensor=weights_scaled, mask=invalid_probs_weights)

            # def bias_print():
            #     tf.print("HACK WARNING: if the following is NOT empty, your sampling _may_ be biased."
            #              " Failed weights:", failed_weights, " failed probs", failed_probs)

            # tf.cond(tf.not_equal(tf.shape(input=failed_weights), [0]), bias_print, lambda: None)

            tf.debugging.assert_equal(tf.shape(input=failed_weights), [0])

            # for weights scaled more then ratio_threshold
            if run.numeric_checks:
                tf.debugging.assert_greater_equal(
                    x=weights_scaled,
                    y=probabilities,
                    message="Not all weights are >= probs so the sampling "
                    "will be biased. If a custom `sample_and_weights` "
                    "was used, make sure that either the shape of the "
                    "custom sampler (resp. it's weights) overlap better "
                    "or decrease the `max_weight`",
                )
            #
            # # check disabled (below not added to deps)
            # assert_scaling_op = tf.assert_less(weights_scaling / min_prob_weights_ratio, z.constant(ratio_threshold),
            #                                    data=[weights_scaling, min_prob_weights_ratio],
            #                                    message="The ratio between the probabilities from the pdf and the"
            #                                    f"probability from the sampler is higher "
            #                                    f" then {ratio_threshold}. This will most probably bias the sampling. "
            #                                    f"Use importance sampling or, to disable this check, do"
            #                                    f"zfit.run.numeric_checks = False")
            # assert_op.append(assert_scaling_op)

        take_or_not = probabilities > random_thresholds
        take_or_not = take_or_not[0] if len(take_or_not.shape) == 2 else take_or_not
        filtered_sample = tf.boolean_mask(tensor=rnd_sample, mask=take_or_not, axis=0)

        n_accepted = tf.shape(input=filtered_sample, out_type=tf.int64)[0]
        n_produced_new = n_produced + n_accepted
        if not dynamic_array_shape:
            indices = tf.boolean_mask(tensor=draw_indices, mask=take_or_not)
            current_sampled = tf.sparse.to_dense(
                tf.SparseTensor(
                    indices=indices,
                    values=tf.broadcast_to(input=(True,), shape=(n_accepted,)),
                    dense_shape=(znp.asarray(n, dtype=tf.int64),),
                ),
                default_value=False,
            )
            is_sampled = tf.logical_or(is_sampled, current_sampled)
            indices = indices[:, 0]
        else:
            indices = tf.range(n_produced, n_produced_new)

        # TODO: pack into tf.function to speedup considerable the eager sampling? Is bottleneck currently
        sample_new = sample.scatter(indices=znp.asarray(indices, dtype=tf.int32), value=filtered_sample)

        # efficiency (estimate) of how many samples we get
        eff = znp.max([z.to_real(n_produced_new), z.to_real(0.5)]) / znp.max([z.to_real(n_total_drawn), z.to_real(1.0)])
        return (
            n,
            sample_new,
            n_produced_new,
            n_total_drawn,
            eff,
            is_sampled,
            weights_scaling,
            weights_maximum,
            prob_maximum,
            n_min_to_produce,
        )

    efficiency_estimation = z.to_real(efficiency_estimation)
    weights_scaling = z.constant(0.0)
    weights_maximum = z.constant(0.0)
    prob_maximum = z.constant(0.0)
    n_min_to_produce = znp.asarray(n_min_to_produce, dtype=tf.int64)
    inital_n_produced = znp.asarray(inital_n_produced, dtype=tf.int64)
    initial_n_drawn = znp.asarray(initial_n_drawn, dtype=tf.int64)

    loop_vars = (
        n,
        sample,
        inital_n_produced,
        initial_n_drawn,
        efficiency_estimation,
        initial_is_sampled,
        weights_scaling,
        weights_maximum,
        prob_maximum,
        n_min_to_produce,
    )

    sample_array = tf.while_loop(
        cond=not_enough_produced,
        body=sample_body,  # paraopt
        loop_vars=loop_vars,
        swap_memory=True,
        parallel_iterations=1,
    )[1]
    new_sample = sample_array.stack()
    new_sample = tf.stop_gradient(new_sample)  # stopping backprop

    if multiple_limits:
        new_sample = z.random.shuffle(new_sample)  # to make sure, randomly remove and not biased.
    if dynamic_array_shape:  # if not dynamic we produced exact n -> no need to cut
        new_sample = new_sample[:n, :]  # cutting away to many produced

    # if no failure, uncomment both for improvement of shape inference, but what if n is tensor?
    if run.executing_eagerly():
        n_dims = limits.n_obs
        with suppress(AttributeError):  # if n_samples_int is not a numpy object
            new_sample.set_shape((int(n), n_dims))
    return new_sample


def extract_extended_pdfs(pdfs: Iterable[ZfitPDF] | ZfitPDF) -> list[ZfitPDF]:
    """Return all extended pdfs that are daughters.

    Args:
        pdfs:

    Returns:
        List[pdfs]:
    """
    from ..models.functor import BaseFunctor

    pdfs = convert_to_container(pdfs)
    indep_pdfs = []

    for pdf in pdfs:
        if not pdf.is_extended:
            continue
        if isinstance(pdf, BaseFunctor):
            if all(pdf.pdfs_extended):
                indep_pdfs.extend(extract_extended_pdfs(pdfs=pdf.pdfs))
            elif not any(pdf.pdfs_extended):
                indep_pdfs.append(pdf)
            else:
                msg = "Should not reach this point, wrong assumptions. Please report bug."
                raise AssertionError(msg)
        else:  # extended, but not a functor
            indep_pdfs.append(pdf)

    return indep_pdfs


def extended_sampling(pdfs: Iterable[ZfitPDF] | ZfitPDF, limits: Space) -> tf.Tensor:
    """Create a sample from extended pdfs by sampling from a Poisson using the yield.

    Args:
        pdfs:
        limits:

    Returns:
        Union[tf.Tensor]:
    """
    samples = []
    pdfs = convert_to_container(pdfs)
    pdfs = extract_extended_pdfs(pdfs)

    for pdf in pdfs:
        n = z.random.poisson(lam=pdf.get_yield(), shape=(), dtype=ztypes.float)
        n = znp.asarray(n, dtype=tf.int64)
        sample = pdf.sample(limits=limits, n=n)
        # sample.set_shape((n, limits.n_obs))
        samples.append(sample.value())

    return znp.concatenate(samples, axis=0)
