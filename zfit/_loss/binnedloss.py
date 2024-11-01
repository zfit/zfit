#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from uhi.typing.plottable import PlottableHistogram

from .. import z
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitParameter
from ..core.loss import BaseLoss
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from ..util.warnings import warn_advanced_feature
from ..util.ztyping import ConstraintsInputType, OptionsInputType
from ..z import numpy as znp


@z.function(wraps="tensor", keepalive=True)
def _spd_transform(values, probs, variances):
    """Transform the data to the SPD form.

    Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6

    The scale is >= 1 to ensure that empty data bins are handled correctly.

    Args:
        values: Data values, the counts in each bin.
        probs: Data probabilities, the expected counts in each bin.
        variances: Data variances, the variances of the counts in each bin.

    Returns:
        The transformed probabilities and values.
    """
    # Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    scale = znp.maximum(values * tf.math.reciprocal_no_nan(variances), znp.ones_like(values))
    probs = probs * scale
    values = values * scale
    return probs, values


@z.function(wraps="tensor", keepalive=True)
def poisson_loss_calc(probs, values, log_offset=None, variances=None):
    """Calculate the Poisson log probability for a given set of data.

    Args:
        probs: Probabilities of the data, i.e., the expected number of events in each bin.
        values: Values of the data, i.e., the number of events in each bin.
        log_offset: Optional offset to be added to the loss. Useful for adding a constant to the loss to improve
            numerical stability.
        variances: (currently ignored)
    Returns:
        The Poisson log probability for the given data.
    """
    # Optional variances of the data. If not None, the Poisson loss is calculated using the
    #             scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    if log_offset is None:
        log_offset = False
    use_offset = log_offset is not False
    if False:  # TODO: this gives very different uncertainties?  if fixed, rechange the docs
        values, probs = _spd_transform(values, probs, variances=variances)
    values += znp.asarray(1e-307, dtype=znp.float64)
    probs += znp.asarray(1e-307, dtype=znp.float64)
    poisson_term = tf.nn.log_poisson_loss(
        values,
        znp.log(probs),
        compute_full_loss=not use_offset,  # TODO: correct offset
    )  # TODO: optimization?

    # cross-check
    # import tensorflow_probability as tfp
    # poisson_dist = tfp.distributions.Poisson(rate=probs)
    # poisson_term = -poisson_dist.log_prob(values)
    if use_offset:
        log_offset = znp.asarray(log_offset, dtype=znp.float64)
        poisson_term += log_offset
    return poisson_term


class BaseBinned(BaseLoss):
    def __init__(
        self,
        model: ztyping.BinnedPDFInputType,
        data: ztyping.BinnedDataInputType,
        constraints: ConstraintsInputType = None,
        options: OptionsInputType = None,
    ):
        model = convert_to_container(model)
        data = convert_to_container(data)
        from zfit._data.binneddatav1 import BinnedData

        data = [
            (
                BinnedData.from_hist(d)
                if (isinstance(d, PlottableHistogram) and not isinstance(d, ZfitBinnedData))
                else d
            )
            for d in data
        ]
        not_binned_pdf = [mod for mod in model if not isinstance(mod, ZfitBinnedPDF)]
        not_binned_data = [dat for dat in data if not isinstance(dat, ZfitBinnedData)]
        not_binned_pdf_msg = (
            "The following PDFs are not binned but need to be. They can be wrapped in an "
            f"BinnedFromUnbinnedPDF. {not_binned_pdf} "
        )
        not_binned_data_msg = (
            "The following datasets are not binned but need to be. They can be converted to a binned "
            f"using the `to_binned` method. {not_binned_data}"
        )
        error_msg = ""
        if not_binned_pdf:
            error_msg += not_binned_pdf_msg
        if not_binned_data:
            error_msg += not_binned_data_msg
        if error_msg:
            raise ValueError(error_msg)

        super().__init__(
            model=model,
            data=data,
            constraints=constraints,
            fit_range=None,
            options=options,
        )

    def create_new(
        self,
        model: ztyping.BinnedPDFInputType = NONE,
        data: ztyping.BinnedDataInputType = NONE,
        constraints: ConstraintsInputType = NONE,
        options: OptionsInputType = NONE,
    ):
        r"""Create a new binned loss of this type. This is preferrable over creating a new instance in most cases.

        Internals, such as certain optimizations will be shared and therefore the loss is made comparable.

        If something is not given, it will be taken from the current loss.

        Args:
            model: |@doc:loss.binned.init.model| Binned PDF(s) that return the normalized probability
               (``rel_counts`` or ``counts``) for
               *data* under the given parameters.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.model|
            data: |@doc:loss.binned.init.data| Binned dataset that will be given to the *model*.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.data|
            constraints: |@doc:loss.init.constraints| Auxiliary measurements ("constraints")
               that add a likelihood term to the loss.

               .. math::
                 \mathcal{L}(\theta) = \mathcal{L}_{unconstrained} \prod_{i} f_{constr_i}(\theta)

               Usually, an auxiliary measurement -- by its very nature -S  should only be added once
               to the loss. zfit does not automatically deduplicate constraints if they are given
               multiple times, leaving the freedom for arbitrary constructs.

               Constraints can also be used to restrict the loss by adding any kinds of penalties. |@docend:loss.init.constraints|
            options: |@doc:loss.init.options| Additional options (as a dict) for the loss.
               Current possibilities include:

               - 'subtr_const' (default True): subtract from each points
                 log probability density a constant that
                 is approximately equal to the average log probability
                 density in the very first evaluation before
                 the summation. This brings the initial loss value closer to 0 and increases,
                 especially for large datasets, the numerical stability.

                 The value will be stored ith 'subtr_const_value' and can also be given
                 directly.

                 The subtraction should not affect the minimum as the absolute
                 value of the NLL is meaningless. However,
                 with this switch on, one cannot directly compare
                 different likelihoods absolute value as the constant
                 may differ! Use ``create_new`` in order to have a comparable likelihood
                 between different losses or use the ``full`` argument in the value function
                 to calculate the full loss with all constants.


               These settings may extend over time. In order to make sure that a loss is the
               same under the same data, make sure to use ``create_new`` instead of instantiating
               a new loss as the former will automatically overtake any relevant constants
               and behavior. |@docend:loss.init.options|

        Returns:
        """
        if model is NONE:
            model = self.model
        if data is NONE:
            data = self.data
        if constraints is NONE:
            constraints = self.constraints
            if constraints is not None:
                constraints = constraints.copy()
        if options is NONE:
            options = self._options
            if isinstance(options, dict):
                options = options.copy()
        return type(self)(model=model, data=data, constraints=constraints, options=options)


class ExtendedBinnedNLL(BaseBinned):
    def __init__(
        self,
        model: ztyping.BinnedPDFInputType,
        data: ztyping.BinnedDataInputType,
        constraints: ConstraintsInputType = None,
        options: OptionsInputType = None,
    ):
        r"""Extended binned likelihood using the expected number of events per bin with a poisson probability.

            The binned likelihood is defined as

            .. math::
                \mathcal{L} = \product \mathcal{poiss}(N_{modelbin_i}, N_{databin_i})
                = N_{databin_i}^{N_{modelbin_i}} \frac{e^{- N_{databin_i}}}{N_{modelbin_i}!}


            where :math:`databin_i` is the :math:`i^{th}` bin in the data and
            :math:`modelbin_i` is the :math:`i^{th}` bin of the model, the expected counts.

            |@doc:loss.init.explain.simultaneous| A simultaneous fit can be performed by giving one or more ``model``, ``data``, to the loss. The
        length of each has to match the length of the others

        .. math::
            \mathcal{L}_{simultaneous}(\theta | {data_0, data_1, ..., data_n})
            = \prod_{i} \mathcal{L}(\theta_i, data_i)

        where :math:`\theta_i` is a set of parameters and
        a subset of :math:`\theta` |@docend:loss.init.explain.simultaneous|

            |@doc:loss.init.explain.negativelog| For optimization purposes, it is often easier
        to minimize a function and to use a log transformation. The actual loss is given by

        .. math::
             \mathcal{L} = - \sum_{i}^{n} ln(f(\theta|x_i))

        and therefore being called "negative log ..." |@docend:loss.init.explain.negativelog|

            Args:
                model: |@doc:loss.binned.init.model| Binned PDF(s) that return the normalized probability
               (``rel_counts`` or ``counts``) for
               *data* under the given parameters.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.model|
                data: |@doc:loss.binned.init.data| Binned dataset that will be given to the *model*.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.data|
                constraints: |@doc:loss.init.constraints| Auxiliary measurements ("constraints")
               that add a likelihood term to the loss.

               .. math::
                 \mathcal{L}(\theta) = \mathcal{L}_{unconstrained} \prod_{i} f_{constr_i}(\theta)

               Usually, an auxiliary measurement -- by its very nature -S  should only be added once
               to the loss. zfit does not automatically deduplicate constraints if they are given
               multiple times, leaving the freedom for arbitrary constructs.

               Constraints can also be used to restrict the loss by adding any kinds of penalties. |@docend:loss.init.constraints|
                options: |@doc:loss.init.options| Additional options (as a dict) for the loss.
               Current possibilities include:

               - 'subtr_const' (default True): subtract from each points
                 log probability density a constant that
                 is approximately equal to the average log probability
                 density in the very first evaluation before
                 the summation. This brings the initial loss value closer to 0 and increases,
                 especially for large datasets, the numerical stability.

                 The value will be stored ith 'subtr_const_value' and can also be given
                 directly.

                 The subtraction should not affect the minimum as the absolute
                 value of the NLL is meaningless. However,
                 with this switch on, one cannot directly compare
                 different likelihoods absolute value as the constant
                 may differ! Use ``create_new`` in order to have a comparable likelihood
                 between different losses or use the ``full`` argument in the value function
                 to calculate the full loss with all constants.


               These settings may extend over time. In order to make sure that a loss is the
               same under the same data, make sure to use ``create_new`` instead of instantiating
               a new loss as the former will automatically overtake any relevant constants
               and behavior. |@docend:loss.init.options|
        """

        # readd below if fixed
        #     |@doc:loss.init.explain.spdtransform| A scaled Poisson
        self._errordef = 0.5
        super().__init__(model=model, data=data, constraints=constraints, options=options)

    @z.function(wraps="loss", keepalive=True)
    def _loss_func(
        self,
        model: Iterable[ZfitBinnedPDF],
        data: Iterable[ZfitBinnedData],
        fit_range,
        constraints,
        log_offset,
    ):
        del fit_range
        poisson_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            probs = mod.counts(dat)
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            poisson_term_summed = znp.sum(poisson_term)
            poisson_terms.append(poisson_term_summed)  # TODO: change None
        nll = znp.sum(poisson_terms)

        if constraints:
            log_offset_val = (
                0.0 if log_offset is False else log_offset
            )  # we need to check identity, cannot do runtime conditional if jitted
            log_offset_val = znp.asarray(log_offset_val, dtype=znp.float64)
            constraints = z.reduce_sum([c.value() - log_offset_val * len(c.get_params()) for c in constraints])
            nll += constraints

        return nll

    @property
    def is_extended(self):
        return True

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        return super()._get_params(
            floating=floating, is_yield=is_yield, extract_independent=extract_independent, autograd=autograd
        )


class BinnedNLL(BaseBinned):
    def __init__(
        self,
        model: ztyping.BinnedPDFInputType,
        data: ztyping.BinnedDataInputType,
        constraints: ConstraintsInputType = None,
        options: OptionsInputType = None,
    ):
        r"""Binned negative log likelihood.

            The binned likelihood is the binned version of :py:class:`~zfit.loss.UnbinnedNLL`. It is defined as

            .. math::
                \\mathcal{L} = \\product \\mathcal{poiss}(N_{modelbin_i}, N_{databin_i}) = N_{databin_i}^{N_{modelbin_i}} \frac{e^{- N_{databin_i}}}{N_{modelbin_i}!}


            where :math:`databin_i` is the :math:`i^{th}` bin in the data and
            :math:`modelbin_i` is the :math:`i^{th}` bin of the model multiplied by the total number of events in data.

            |@doc:loss.init.explain.simultaneous| A simultaneous fit can be performed by giving one or more ``model``, ``data``, to the loss. The
        length of each has to match the length of the others

        .. math::
            \mathcal{L}_{simultaneous}(\theta | {data_0, data_1, ..., data_n})
            = \prod_{i} \mathcal{L}(\theta_i, data_i)

        where :math:`\theta_i` is a set of parameters and
        a subset of :math:`\theta` |@docend:loss.init.explain.simultaneous|

            |@doc:loss.init.explain.negativelog| For optimization purposes, it is often easier
        to minimize a function and to use a log transformation. The actual loss is given by

        .. math::
             \mathcal{L} = - \sum_{i}^{n} ln(f(\theta|x_i))

        and therefore being called "negative log ..." |@docend:loss.init.explain.negativelog|

            Args:
                model: |@doc:loss.binned.init.model| Binned PDF(s) that return the normalized probability
               (``rel_counts`` or ``counts``) for
               *data* under the given parameters.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.model|
                data: |@doc:loss.binned.init.data| Binned dataset that will be given to the *model*.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.data|
                constraints: |@doc:loss.init.constraints| Auxiliary measurements ("constraints")
               that add a likelihood term to the loss.

               .. math::
                 \mathcal{L}(\theta) = \mathcal{L}_{unconstrained} \prod_{i} f_{constr_i}(\theta)

               Usually, an auxiliary measurement -- by its very nature -S  should only be added once
               to the loss. zfit does not automatically deduplicate constraints if they are given
               multiple times, leaving the freedom for arbitrary constructs.

               Constraints can also be used to restrict the loss by adding any kinds of penalties. |@docend:loss.init.constraints|
                options: |@doc:loss.init.options| Additional options (as a dict) for the loss.
               Current possibilities include:

               - 'subtr_const' (default True): subtract from each points
                 log probability density a constant that
                 is approximately equal to the average log probability
                 density in the very first evaluation before
                 the summation. This brings the initial loss value closer to 0 and increases,
                 especially for large datasets, the numerical stability.

                 The value will be stored ith 'subtr_const_value' and can also be given
                 directly.

                 The subtraction should not affect the minimum as the absolute
                 value of the NLL is meaningless. However,
                 with this switch on, one cannot directly compare
                 different likelihoods absolute value as the constant
                 may differ! Use ``create_new`` in order to have a comparable likelihood
                 between different losses or use the ``full`` argument in the value function
                 to calculate the full loss with all constants.


               These settings may extend over time. In order to make sure that a loss is the
               same under the same data, make sure to use ``create_new`` instead of instantiating
               a new loss as the former will automatically overtake any relevant constants
               and behavior. |@docend:loss.init.options|
        """

        # readd below if fixed
        #            |@doc:loss.init.explain.spdtransform| A scaled Poisson distribution is...
        self._errordef = 0.5
        super().__init__(model=model, data=data, constraints=constraints, options=options)
        extended_pdfs = [pdf for pdf in self.model if pdf.is_extended]
        if extended_pdfs and type(self) is BinnedNLL:
            warn_advanced_feature(
                f"Extended PDFs ({extended_pdfs}) are given to a normal BinnedNLL. "
                f" This won't take the yield "
                "into account and simply treat the PDFs as non-extended PDFs. To create an "
                "extended NLL, use the `ExtendedBinnedNLL`.",
                identifier="extended_in_BinnedNLL",
            )

    @z.function(wraps="loss", keepalive=True)
    def _loss_func(
        self,
        model: Iterable[ZfitBinnedPDF],
        data: Iterable[ZfitBinnedData],
        fit_range,
        constraints,
        log_offset,
    ):
        del fit_range
        poisson_terms = []
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            variances = dat.variances()
            probs = mod.rel_counts(dat)
            probs *= znp.sum(values)
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            poisson_term_summed = znp.sum(poisson_term)
            poisson_terms.append(poisson_term_summed)
        nll = znp.sum(poisson_terms)

        if constraints:
            log_offset_val = (
                0.0 if log_offset is False else log_offset
            )  # we need to check identity, cannot do runtime conditional if jitted
            log_offset_val = znp.asarray(log_offset_val, dtype=znp.float64)
            constraints = z.reduce_sum([c.value() - log_offset_val * len(c.get_params()) for c in constraints])
            nll += constraints

        return nll

    @property
    def is_extended(self):
        return False

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent, autograd=autograd)


@z.function(wraps="tensor", keepalive=True)
def chi2_loss_calc(probs, values, variances, log_offset=None, ignore_empty=None):
    """Calculate the chi2 for a given set of data.

    Args:
        probs: Probabilities of the data, i.e. the expected number of events in each bin.
        values: Values of the data, i.e. the number of events in each
            bin.
        variances: Data variances, the variances of the counts in each bin.
        log_offset: Optional offset to be added to the loss. Useful for adding a constant to the loss to improve
            numerical stability.
        ignore_empty: If True, empty bins are ignored.
    Returns:
        The chi2 for the given data.
    """

    if log_offset is None:
        log_offset = False
    if ignore_empty is None:
        ignore_empty = True
    chi2_term = tf.math.squared_difference(probs, values)
    one_over_var = tf.math.reciprocal_no_nan(variances) if ignore_empty else tf.math.reciprocal(variances)
    chi2_term *= one_over_var
    if log_offset is not False:
        log_offset = znp.asarray(log_offset, dtype=znp.float64)
        chi2_term += log_offset
    return znp.sum(chi2_term)


def _check_small_counts_chi2(data, ignore_empty):
    for dat in data:
        variances = dat.variances()
        smaller_than_six = dat.values() < 6
        if variances is None:
            msg = f"variances cannot be None for Chi2: {dat}"
            raise ValueError(msg)
        if np.any(variances <= 0) and not ignore_empty:
            msg = f"Variances of {dat} contains zeros or negative numbers, cannot calculate chi2." f" {variances}"
            raise ValueError(msg)
        if np.any(smaller_than_six):
            warn_advanced_feature(
                f"Some values in {dat} are < 6, the chi2 assumption of gaussian distributed"
                f" uncertainties most likely won't hold anymore. Use Chi2 for large samples."
                f"For smaller samples, consider using (Extended)BinnedNLL (or an unbinned fit).",
                identifier="chi2_counts_small",
            )


class BinnedChi2(BaseBinned):
    def __init__(
        self,
        model: ztyping.BinnedPDFInputType,
        data: ztyping.BinnedDataInputType,
        constraints: ConstraintsInputType = None,
        options: OptionsInputType = None,
    ):
        r"""Binned Chi2 loss, using the :math:`N_{tot} from the data.

            .. math::
                \chi^2 = \sum_{\mathrm{bins}} \left( \frac{N_\mathrm{PDF,bin} - N_\mathrm{Data,bin}}{\sigma_\mathrm{Data,bin}} \right)^2

            where

            .. math::
                N_\mathrm{PDF,bin} = \mathrm{pdf}(\text{integral}) \cdot N_\mathrm{Data,tot}
                \sigma_\mathrm{bin} = \text{variance}

            with `variance` the value of :class:`~zfit.data.BinnedData.variances` of the binned data.

            |@doc:loss.init.binned.explain.chi2zeros| If the dataset has empty bins, the errors
        will be zero and :math:`\chi^2` is undefined. Two possibilities are available and
        can be given as an option:

        - "empty": "ignore" will ignore all bins with zero entries and won't count to the loss
        - "errors": "expected" will use the expected counts from the model
          with a Poissonian uncertainty |@docend:loss.init.binned.explain.chi2zeros|

            Args:
                model: |@doc:loss.binned.init.model| Binned PDF(s) that return the normalized probability
               (``rel_counts`` or ``counts``) for
               *data* under the given parameters.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.model|
                data: |@doc:loss.binned.init.data| Binned dataset that will be given to the *model*.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.data|
                constraints: |@doc:loss.init.constraints| Auxiliary measurements ("constraints")
               that add a likelihood term to the loss.

               .. math::
                 \mathcal{L}(\theta) = \mathcal{L}_{unconstrained} \prod_{i} f_{constr_i}(\theta)

               Usually, an auxiliary measurement -- by its very nature -S  should only be added once
               to the loss. zfit does not automatically deduplicate constraints if they are given
               multiple times, leaving the freedom for arbitrary constructs.

               Constraints can also be used to restrict the loss by adding any kinds of penalties. |@docend:loss.init.constraints|
                options: |@doc:loss.init.options| Additional options (as a dict) for the loss.
               Current possibilities include:

               - 'subtr_const' (default True): subtract from each points
                 log probability density a constant that
                 is approximately equal to the average log probability
                 density in the very first evaluation before
                 the summation. This brings the initial loss value closer to 0 and increases,
                 especially for large datasets, the numerical stability.

                 The value will be stored ith 'subtr_const_value' and can also be given
                 directly.

                 The subtraction should not affect the minimum as the absolute
                 value of the NLL is meaningless. However,
                 with this switch on, one cannot directly compare
                 different likelihoods absolute value as the constant
                 may differ! Use ``create_new`` in order to have a comparable likelihood
                 between different losses or use the ``full`` argument in the value function
                 to calculate the full loss with all constants.


               These settings may extend over time. In order to make sure that a loss is the
               same under the same data, make sure to use ``create_new`` instead of instantiating
               a new loss as the former will automatically overtake any relevant constants
               and behavior. |@docend:loss.init.options|
        """
        self._errordef = 1.0
        if options is None:
            options = {}
        if options.get("empty") is None:
            options["empty"] = "ignore"
        if options.get("errors") is None:
            options["errors"] = "data"
        super().__init__(model=model, data=data, constraints=constraints, options=options)
        extended_pdfs = [pdf for pdf in self.model if pdf.is_extended]
        if extended_pdfs and type(self) is BinnedChi2:
            warn_advanced_feature(
                f"Extended PDFs ({extended_pdfs}) are given to a normal BinnedChi2. "
                f" This won't take the yield "
                "into account and simply treat the PDFs as non-extended PDFs. To create an "
                "extended loss, use the `ExtendedBinnedChi2`.",
                identifier="extended_in_BinnedChi2",
            )

    def check_precompile(self, *, params=None, force=False):
        params, needs_compile = super().check_precompile(force=force, params=params)
        if needs_compile:
            ignore_empty = self._options.get("empty") == "ignore" or self._options.get("errors") == "expected"

            data = self.data
            _check_small_counts_chi2(data, ignore_empty)
        return params, needs_compile

    @z.function(wraps="loss", keepalive=True)
    def _loss_func(
        self,
        model: Iterable[ZfitBinnedPDF],
        data: Iterable[ZfitBinnedData],
        fit_range,
        constraints,
        log_offset,
    ):
        del fit_range
        ignore_empty = self._options.get("empty") == "ignore"
        chi2_terms = []
        log_offset_val = 0.0 if log_offset is False else log_offset
        log_offset_val = znp.asarray(log_offset_val, dtype=znp.float64)

        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            probs = mod.rel_counts(dat)
            probs *= znp.sum(values)

            variance_method = self._options.get("errors")
            if variance_method == "expected":
                variances = znp.sqrt(probs + znp.asarray(1e-307, dtype=znp.float64))
            elif variance_method == "data":
                variances = dat.variances()
            else:
                raise ValueError()
            if variances is None:
                msg = f"variances cannot be None for Chi2: {dat}"
                raise ValueError(msg)

            chi2_term = chi2_loss_calc(probs, values, variances, log_offset, ignore_empty=ignore_empty)
            chi2_terms.append(chi2_term)
        chi2_term = znp.sum(chi2_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() - log_offset_val for c in constraints])
            chi2_term += constraints

        return chi2_term

    @property
    def is_extended(self):
        return False

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent, autograd=autograd)


class ExtendedBinnedChi2(BaseBinned):
    def __init__(
        self,
        model: ztyping.BinnedPDFInputType,
        data: ztyping.BinnedDataInputType,
        constraints: ConstraintsInputType = None,
        options: OptionsInputType = None,
    ):
        r"""Binned Chi2 loss, using the :math:`N_{tot} from the PDF.

            .. math::
                \chi^2 = \sum_{\mathrm{bins}} \left( \frac{N_\mathrm{PDF,bin} - N_\mathrm{Data,bin}}{\sigma_\mathrm{Data,bin}} \right)^2

            where

            .. math::
                N_\mathrm{PDF,bin} = \mathrm{pdf}(\text{integral}) \cdot N_\mathrm{PDF,expected}
                \sigma_\mathrm{bin} = \text{variance}

            with `variance` the value of :class:`~zfit.data.BinnedData.variances` of the binned data.

            |@doc:loss.init.binned.explain.chi2zeros| If the dataset has empty bins, the errors
        will be zero and :math:`\chi^2` is undefined. Two possibilities are available and
        can be given as an option:

        - "empty": "ignore" will ignore all bins with zero entries and won't count to the loss
        - "errors": "expected" will use the expected counts from the model
          with a Poissonian uncertainty |@docend:loss.init.binned.explain.chi2zeros|


        Args:
            model: |@doc:loss.binned.init.model| Binned PDF(s) that return the normalized probability
               (``rel_counts`` or ``counts``) for
               *data* under the given parameters.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.model|
                data: |@doc:loss.binned.init.data| Binned dataset that will be given to the *model*.
               If multiple model and data are given, they will be used
               in the same order to do a simultaneous fit. |@docend:loss.binned.init.data|
                constraints: |@doc:loss.init.constraints| Auxiliary measurements ("constraints")
               that add a likelihood term to the loss.

               .. math::
                 \mathcal{L}(\theta) = \mathcal{L}_{unconstrained} \prod_{i} f_{constr_i}(\theta)

               Usually, an auxiliary measurement -- by its very nature -S  should only be added once
               to the loss. zfit does not automatically deduplicate constraints if they are given
               multiple times, leaving the freedom for arbitrary constructs.

               Constraints can also be used to restrict the loss by adding any kinds of penalties. |@docend:loss.init.constraints|
                options: |@doc:loss.init.options| Additional options (as a dict) for the loss.
               Current possibilities include:

               - 'subtr_const' (default True): subtract from each points
                 log probability density a constant that
                 is approximately equal to the average log probability
                 density in the very first evaluation before
                 the summation. This brings the initial loss value closer to 0 and increases,
                 especially for large datasets, the numerical stability.

                 The value will be stored ith 'subtr_const_value' and can also be given
                 directly.

                 The subtraction should not affect the minimum as the absolute
                 value of the NLL is meaningless. However,
                 with this switch on, one cannot directly compare
                 different likelihoods absolute value as the constant
                 may differ! Use ``create_new`` in order to have a comparable likelihood
                 between different losses or use the ``full`` argument in the value function
                 to calculate the full loss with all constants.


               These settings may extend over time. In order to make sure that a loss is the
               same under the same data, make sure to use ``create_new`` instead of instantiating
               a new loss as the former will automatically overtake any relevant constants
               and behavior. |@docend:loss.init.options|
        """
        self._errordef = 1.0
        if options is None:
            options = {}
        if options.get("empty") is None:
            options["empty"] = "ignore"
        if options.get("errors") is None:
            options["errors"] = "data"
        super().__init__(model=model, data=data, constraints=constraints, options=options)

    def check_precompile(self, *, params=None, force=None):
        params, needs_compile = super().check_precompile(params=params, force=force)
        if needs_compile:
            ignore_empty = self._options.get("empty") == "ignore" or self._options.get("errors") == "expected"

            data = self.data
            _check_small_counts_chi2(data, ignore_empty)
        return params, needs_compile

    @z.function(wraps="loss", keepalive=True)
    def _loss_func(
        self,
        model: Iterable[ZfitBinnedPDF],
        data: Iterable[ZfitBinnedData],
        fit_range,
        constraints,
        log_offset,
    ):
        del fit_range
        ignore_empty = self._options.get("empty") == "ignore"
        chi2_terms = []
        log_offset_val = 0.0 if log_offset is False else log_offset
        log_offset_val = znp.asarray(log_offset_val, dtype=znp.float64)
        for mod, dat in zip(model, data):
            values = dat.values(  # TODO: right order of model and data?
                # obs=mod.obs
            )
            probs = mod.counts(dat)
            variance_method = self._options.get("errors")
            if variance_method == "expected":
                variances = znp.sqrt(probs + znp.asarray(1e-307, dtype=znp.float64))
            elif variance_method == "data":
                variances = dat.variances()
            else:
                msg = f"Variance method {variance_method} not supported"
                raise ValueError(msg)
            if variances is None:
                msg = f"variances cannot be None for Chi2: {dat}"
                raise ValueError(msg)

            chi2_term = chi2_loss_calc(probs, values, variances, log_offset, ignore_empty=ignore_empty)
            chi2_terms.append(chi2_term)
        chi2_term = znp.sum(chi2_terms)

        if constraints:
            constraints = z.reduce_sum([c.value() - log_offset_val for c in constraints])
            chi2_term += constraints

        return chi2_term

    @property
    def is_extended(self):
        return True
