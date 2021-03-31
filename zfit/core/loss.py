#  Copyright (c) 2021 zfit

import abc
import inspect
import warnings
from typing import (Callable, Iterable, List, Mapping, Optional, Set, Tuple,
                    Union)

import tensorflow as tf
from ordered_set import OrderedSet

from .. import settings, z
from .interfaces import ZfitPDF

znp = z.numpy
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container, is_container
from ..util.deprecation import deprecated, deprecated_args
from ..util.exception import (BreakingAPIChangeError, IntentionAmbiguousError,
                              NotExtendedPDFError, WorkInProgressError)
from ..util.warnings import warn_advanced_feature
from ..z.math import (autodiff_gradient, autodiff_value_gradients,
                      automatic_value_gradients_hessian, numerical_gradient,
                      numerical_value_gradient,
                      numerical_value_gradients_hessian)
from .baseobject import BaseNumeric
from .constraint import BaseConstraint
from .dependents import _extract_dependencies
from .interfaces import ZfitData, ZfitLoss, ZfitSpace
from .parameter import convert_to_parameters


# @z.function
def _unbinned_nll_tf(model: ztyping.PDFInputType, data: ztyping.DataInputType, fit_range: ZfitSpace, log_offset=None):
    """Return unbinned negative log likelihood graph for a PDF.

    Args:
        model: PDFs with a `.pdf` method. Has to be as many models as data
        data:
        fit_range:

    Returns:
        The unbinned nll

    Raises:
        ValueError: if both `probs` and `log_probs` are specified.
    """

    if is_container(model):
        nlls = [_unbinned_nll_tf(model=p, data=d, fit_range=r, log_offset=log_offset)
                for p, d, r in zip(model, data, fit_range)]
        # nlls_total = [nll.total for nll in nlls]
        # nlls_correction = [nll.correction for nll in nlls]
        # nlls_total_summed = tf.reduce_sum(input_tensor=nlls_total, axis=0)
        nlls_summed = tf.reduce_sum(input_tensor=nlls, axis=0)

        # nlls_correction_summed = tf.reduce_sum(input_tensor=nlls_correction, axis=0)
        # nll_finished = (nlls_total_summed, nlls_correction_summed)
        nll_finished = nlls_summed
    else:
        if fit_range is not None:
            with data.set_data_range(fit_range):
                probs = model.pdf(data, norm_range=fit_range)
        else:
            probs = model.pdf(data)
        log_probs = tf.math.log(probs)
        nll = _nll_calc_unbinned_tf(log_probs=log_probs,
                                    weights=data.weights if data.weights is not None else None,
                                    log_offset=log_offset)
        nll_finished = nll
    return nll_finished


@z.function(wraps='tensor')
def _nll_calc_unbinned_tf(log_probs, weights=None, log_offset=None):
    if weights is not None:
        log_probs *= weights  # because it's prob ** weights
    if log_offset is not None:
        log_probs -= log_offset
    nll = -tf.reduce_sum(input_tensor=log_probs, axis=0)
    # nll = -tfp.math.reduce_kahan_sum(input_tensor=log_probs, axis=0)
    return nll


def _constraint_check_convert(constraints):
    checked_constraints = []
    for constr in constraints:
        if isinstance(constr, BaseConstraint):
            checked_constraints.append(constr)
        else:
            raise BreakingAPIChangeError("Constraints have to be of type `Constraint`, a simple"
                                         " constraint from a function can be constructed with"
                                         " `SimpleConstraint`.")
    return checked_constraints


class BaseLoss(ZfitLoss, BaseNumeric):

    def __init__(self, model: ztyping.ModelsInputType, data: ztyping.DataInputType,
                 fit_range: ztyping.LimitsTypeInput = None,
                 constraints: ztyping.ConstraintsTypeInput = None,
                 options: Optional[Mapping] = None):
        # first doc line left blank on purpose, subclass adds class docstring (Sphinx autodoc adds the two)
        """A "simultaneous fit" can be performed by giving one or more `model`, `data`, `fit_range` to the loss. The
        length of each has to match the length of the others.

        Args:
            model: The model or models to evaluate the data on
            data: Data to use
            fit_range: The fitting range. It's the norm_range for the models (if
            they
                have a norm_range) and the data_range for the data.
            constraints: A Tensor representing a loss constraint. Using
                `zfit.constraint.*` allows for easy use of predefined constraints.
            options: Different options for the loss calculation.
        """
        super().__init__(name=type(self).__name__, params={})
        if fit_range is not None:
            warnings.warn("The fit_range argument is depreceated and will maybe removed in future releases. "
                          "It is preferred to define the range in the space"
                          " when creating the data and the model.", stacklevel=2)

        model, data, fit_range = self._input_check(pdf=model, data=data, fit_range=fit_range)
        self._model = model
        self._data = data
        self._fit_range = fit_range

        options = self._check_init_options(options, data)

        self._options = options
        self._subtractions = {}
        if constraints is None:
            constraints = []
        self._constraints = _constraint_check_convert(convert_to_container(constraints, list))

        self._precompile()

    def _check_init_options(self, options, data):
        try:
            nevents = sum(d.nevents for d in data)
        except RuntimeError:  # can happen if not yet sampled. What to do? Approx_nevents?
            nevents = 150_000  # sensible default
        options = {} if options is None else options

        if options.get('numhess') is None:
            options['numhess'] = True

        if options.get('numgrad') is None:
            options['numgrad'] = settings.options['numerical_grad']

        if options.get('kahansum') is None:
            options['kahansum'] = nevents > 500_000  # start using kahan if we have more than 500k events

        if options.get('subtr_const') is None:
            if nevents < 200_000:
                subst_const = True
            elif nevents < 1_000_000:
                subst_const = 'kahan'
            else:
                subst_const = 'elewise'

            options['subtr_const'] = subst_const

        return options

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._name = "UnnamedSubBaseLoss"

    def _get_params(self,
                    floating: Optional[bool] = True,
                    is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = OrderedSet()
        params = params.union(*(model.get_params(floating=floating, is_yield=is_yield,
                                                 extract_independent=extract_independent)
                                for model in self.model))

        params = params.union(*(constraint.get_params(floating=floating, is_yield=False,
                                                      extract_independent=extract_independent)
                                for constraint in self.constraints))
        return params

    def _input_check(self, pdf, data, fit_range):
        if is_container(pdf) ^ is_container(data):
            raise ValueError("`pdf` and `data` either both have to be a list or not.")
        if not is_container(pdf) and isinstance(fit_range, list):
            raise TypeError("`pdf` and `data` are not a `list`, `fit_range` can't be a `list` then.")
        if isinstance(pdf, tuple):
            raise TypeError("`pdf` has to be a pdf or a list of pdfs, not a tuple.")

        if isinstance(data, tuple):
            raise TypeError("`data` has to be a data or a list of data, not a tuple.")

        # pdf, data = (convert_to_container(obj, non_containers=[tuple]) for obj in (pdf, data))
        pdf, data = self._check_convert_model_data(pdf, data, fit_range)
        # TODO: data, range consistency?
        if fit_range is None:
            fit_range = []
            non_consistent = {'data': [], 'model': [], 'range': []}
            for p, d in zip(pdf, data):
                if p.norm_range != d.data_range:
                    non_consistent['data'].append(d)
                    non_consistent['model'].append(p)
                    non_consistent['range'].append((p.space, d.data_range))
                fit_range.append(None)
            if non_consistent['range']:  # TODO: test
                warn_advanced_feature(f"PDFs {non_consistent['model']} as "
                                      f"well as `data` {non_consistent['data']}"
                                      f" have different ranges {non_consistent['range']} they"
                                      f" are defined in. The data range will cut the data while the"
                                      f" norm range defines the normalization.",
                                      identifier='inconsistent_fitrange')
        else:
            fit_range = convert_to_container(fit_range, non_containers=[tuple])

        if not len(pdf) == len(data) == len(fit_range):
            raise ValueError("pdf, data and fit_range don't have the same number of components:"
                             "\npdf: {}"
                             "\ndata: {}"
                             "\nfit_range: {}".format(pdf, data, fit_range))

        # sanitize fit_range
        fit_range = [p.convert_sort_space(limits=range_) if range_ is not None else None for p, range_ in
                     zip(pdf, fit_range)]
        # TODO: sanitize pdf, data?
        self.add_cache_deps(cache_deps=pdf)
        self.add_cache_deps(cache_deps=data)
        return pdf, data, fit_range

    def _precompile(self):
        if self._options['subtr_const']:
            log_offset = self._subtractions.get('subtr_const')
            if log_offset is None:
                from zfit import run
                run.assert_executing_eagerly()  # first time subtr
                nevents_tot = znp.sum([d._approx_nevents for d in self.data])
                log_offset_sum = (self._call_value(data=self.data,
                                                   model=self.model,
                                                   fit_range=self.fit_range,
                                                   constraints=self.constraints,
                                                   # presumably were not at the minimum,
                                                   # so the loss will decrease
                                                   log_offset=z.convert_to_tensor(0.)) - 1000.)
                log_offset = tf.stop_gradient(- znp.divide(log_offset_sum, nevents_tot))
                self._subtractions['subtr_const'] = log_offset

    def _check_convert_model_data(self, model, data, fit_range):
        model, data = tuple(convert_to_container(obj) for obj in (model, data))

        model_checked = []
        data_checked = []
        for mod, dat in zip(model, data):
            if not isinstance(dat, ZfitData):
                if fit_range is not None:
                    raise TypeError("Fit range should not be used if data is not ZfitData.")

                if not isinstance(dat, (tf.Tensor, tf.Variable)):
                    try:
                        dat = z.convert_to_tensor(value=dat)
                    except TypeError:
                        raise TypeError(
                            f"Wrong type of dat ({type(dat)}). Has to be a `ZfitData` or convertible to a tf.Tensor")
                # check dimension
                from zfit import Data
                dat = Data.from_tensor(obs=mod.space, tensor=dat)
            model_checked.append(mod)
            data_checked.append(dat)
        return model_checked, data_checked

    def _input_check_params(self, params):
        if params is None:
            params = list(self.get_params())
        else:
            params = convert_to_container(params)
        return params

    def add_constraints(self, constraints):
        constraints = convert_to_container(constraints)
        return self._add_constraints(constraints)

    def _add_constraints(self, constraints):
        constraints = _constraint_check_convert(convert_to_container(constraints, container=list))
        self._constraints.extend(constraints)
        return constraints

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def fit_range(self):
        return self._fit_range

    @property
    def constraints(self):
        return self._constraints

    def _get_dependencies(self):  # TODO: fix, add constraints
        pdf_dependents = _extract_dependencies(self.model)
        pdf_dependents |= _extract_dependencies(self.constraints)
        return pdf_dependents

    @abc.abstractmethod
    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        raise NotImplementedError

    @property
    def errordef(self) -> Union[float, int]:
        return self._errordef

    def value(self):
        log_offset = self._subtractions.get('subtr_const')

        value = self._call_value(self.model, self.data, self.fit_range, self.constraints, log_offset)
        return value

    def _call_value(self, model, data, fit_range, constraints, log_offset):
        value = self._value(model=model,
                            data=data,
                            fit_range=fit_range,
                            constraints=constraints,
                            log_offset=log_offset)
        # if self._subtractions.get('kahan') is None:
        #     self._subtractions['kahan'] = value
        # value_subtracted = (value[0] - self._subtractions['kahan'][0]) - (
        #         value[1] - self._subtractions['kahan'][1])
        # return value_subtracted
        return value
        # value = value_substracted[0] - value_substracted[1]

    def _value(self, model, data, fit_range, constraints, log_offset):
        # try:
        return self._loss_func(model=model, data=data, fit_range=fit_range,
                               constraints=constraints, log_offset=log_offset)

    # except NotImplementedError as error:
    #     raise NotImplementedError(f"_loss_func not properly defined! error {error}") from error

    def __add__(self, other):
        if not isinstance(other, BaseLoss):
            raise TypeError("Has to be a subclass of `BaseLoss` or overwrite `__add__`.")
        if type(other) != type(self):
            raise ValueError("cannot safely add two different kind of loss.")
        model = self.model + other.model
        data = self.data + other.data
        fit_range = self.fit_range + other.fit_range
        constraints = self.constraints + other.constraints
        return type(self)(
            model=model, data=data, fit_range=fit_range, constraints=constraints
        )

    def gradient(self, params: ztyping.ParamTypeInput = None) -> List[tf.Tensor]:
        params = self._input_check_params(params)
        numgrad = self._options['numgrad']

        return self._gradient(params=params, numgrad=numgrad)

    @deprecated(None, "Use `gradient` instead.")
    def gradients(self, *args, **kwargs):
        return self.gradient(*args, **kwargs)

    # @z.function(wraps='loss')
    def _gradient(self, params, numgrad):
        if numgrad:
            return numerical_gradient(self.value, params=params)

        else:
            return autodiff_gradient(self.value, params=params)

    def value_gradient(self, params: ztyping.ParamTypeInput) -> Tuple[tf.Tensor, tf.Tensor]:
        params = self._input_check_params(params)
        numgrad = self._options['numgrad']
        return self._value_gradient(params=params, numgrad=numgrad)

    @deprecated(None, "Use `value_gradient` instead.")
    def value_gradients(self, *args, **kwargs):
        return self.value_gradient(*args, **kwargs)

    @z.function(wraps='loss')
    def _value_gradient(self, params, numgrad=False):
        if numgrad:
            value, gradient = numerical_value_gradient(self.value, params=params)
        else:
            value, gradient = autodiff_value_gradients(self.value, params=params)
        return value, gradient

    def hessian(self, params: ztyping.ParamTypeInput, hessian=None):
        return self.value_gradient_hessian(params=params, hessian=hessian)[2]

    def value_gradient_hessian(self, params: ztyping.ParamTypeInput, hessian=None, numgrad=None) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
        params = self._input_check_params(params)
        numgrad = self._options['numhess'] if numgrad is None else numgrad
        vals = self._value_gradient_hessian(params=params, hessian=hessian, numerical=numgrad)

        vals = vals[0], z.convert_to_tensor(vals[1]), vals[2]
        return vals

    @deprecated(None, "Use `value_gradient_hessian` instead.")
    def value_gradients_hessian(self, *args, **kwargs):
        return self.value_gradient_hessian(*args, **kwargs)

    @z.function(wraps='loss')
    def _value_gradient_hessian(self, params, hessian, numerical=False):
        if numerical:
            return numerical_value_gradients_hessian(
                func=self.value,
                gradient=self.gradient,
                params=params, hessian=hessian)
        else:
            return automatic_value_gradients_hessian(
                self.value, params=params, hessian=hessian
            )

    def __repr__(self) -> str:
        class_name = repr(self.__class__)[:-2].split(".")[-1]
        return f'<{class_name} ' \
               f'model={one_two_many([model.name for model in self.model])} ' \
               f'data={one_two_many([data.name for data in self.data])} ' \
               f'constraints={one_two_many(self.constraints, many="True")} ' \
               f'>'

    def __str__(self) -> str:
        class_name = repr(self.__class__)[:-2].split(".")[-1]
        return f'<{class_name}' \
               f' model={one_two_many([model for model in self.model])}' \
               f' data={one_two_many([data for data in self.data])}' \
               f' constraints={one_two_many(self.constraints, many="True")}' \
               f'>'


def one_two_many(values, n=3, many='multiple'):
    values = convert_to_container(values)
    if len(values) > n:
        values = many
    return values


class UnbinnedNLL(BaseLoss):
    _name = "UnbinnedNLL"

    def __init__(self, model: Union[ZfitPDF, Iterable[ZfitPDF]],
                 data: Union[ZfitData, Iterable[ZfitData]],
                 fit_range=None,
                 constraints=None, options=None):
        """Unbinned Negative Log Likelihood.

        A simultaneous fit can be performed by giving one or more `model`, `data`, `fit_range` to the loss. The
        length of each has to match the length of the others.

        Args:
            model: The model or models to evaluate the data on
            data: Data to use
            fit_range: The fitting range. It's the norm_range for the models (if
            they have a norm_range) and the data_range for the data.
            constraints: A Tensor representing a loss constraint. Using
                `zfit.constraint.*` allows for easy use of predefined constraints.
            options: Different options for the loss calculation.

              - `subtr_const`, default `True`: subtract from each points log probability density a constant that
                is approximately equal to the average log probability density in the very first evaluation.
                This moves the sum of all components, the actual loss value, closer to 0 which increases the
                numerical stability. This is especially useful for large datasets.

            This should not affect the minimum as the absolute value of the NLL is meaningless. However,
            with this switch on, one cannot directly compare different likelihoods ablolute value as the constant
            may differs!
        """
        super().__init__(model=model, data=data, fit_range=fit_range, constraints=constraints, options=options)
        self._errordef = 0.5
        extended_pdfs = [pdf for pdf in self.model if pdf.is_extended]
        if extended_pdfs and type(self) == UnbinnedNLL:
            warn_advanced_feature("Extended PDFs are given to a normal UnbinnedNLL. This won't take the yield "
                                  "into account and simply treat the PDFs as non-extended PDFs. To create an "
                                  "extended NLL, use the `ExtendedUnbinnedNLL`.", identifier='extended_in_UnbinnedNLL')

    def _loss_func(self, model, data, fit_range, constraints, log_offset):

        return self._loss_func_watched(data=data,
                                       model=model,
                                       fit_range=fit_range,
                                       constraints=constraints,
                                       log_offset=log_offset)

    @property
    def is_extended(self):
        return False

    @z.function(wraps='loss')
    def _loss_func_watched(self, data, model, fit_range, constraints, log_offset):
        nll = _unbinned_nll_tf(model=model, data=data,
                               fit_range=fit_range,
                               log_offset=log_offset)
        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints
        return nll

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent)


class ExtendedUnbinnedNLL(UnbinnedNLL):
    """An Unbinned Negative Log Likelihood with an additional poisson term for the number of events in the dataset."""

    @z.function(wraps='loss')
    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        nll = super()._loss_func(model=model, data=data, fit_range=fit_range, constraints=constraints,
                                 log_offset=log_offset)
        yields = []
        nevents_collected = []
        for mod, dat in zip(model, data):
            if not mod.is_extended:
                raise NotExtendedPDFError(f"The pdf {mod} is not extended but has to be (for an extended fit)")
            nevents = dat.n_events if dat.weights is None else z.reduce_sum(dat.weights)
            nevents = tf.cast(nevents, tf.float64)
            nevents_collected.append(nevents)
            yields.append(mod.get_yield())
        yields = tf.stack(yields, axis=0)
        nevents_collected = tf.stack(nevents_collected, axis=0)

        term_new = tf.nn.log_poisson_loss(nevents_collected, tf.math.log(yields))
        if log_offset is not None:
            term_new += log_offset
        nll += tf.reduce_sum(term_new, axis=0)
        return nll

    @property
    def is_extended(self):
        return True


class SimpleLoss(BaseLoss):
    _name = "SimpleLoss"

    @deprecated_args(None, "Use params instead.", ('deps', 'dependents'))
    def __init__(self,
                 func: Callable,
                 params: Iterable["zfit.Parameter"] = None,
                 errordef: Optional[float] = None,
                 # legacy
                 deps: Iterable["zfit.Parameter"] = NONE,
                 dependents: Iterable["zfit.Parameter"] = NONE,
                 ):
        """Loss from a (function returning a) Tensor.

        This allows for a very generic loss function as the functions only restriction is that is
        should depend on `zfit.Parameter`.

        Args:
            func: Callable that constructs the loss and returns a tensor without taking an argument.
            params: The dependents (independent `zfit.Parameter`) of the loss. Essentially the (free) parameters that
              the `func` depends on.
            errordef: Definition of which change in the loss corresponds to a change of 1 sigma.
                For example, 1 for Chi squared, 0.5 for negative log-likelihood.

        Usage:

        .. code:: python

            import zfit
            from zfit import z

            param1 = zfit.Parameter('param1', 5, 1, 10)
            # we can build a model here if we want, but in principle, it's not necessary

            x = z.random.uniform(shape=(100,))
            y = x * z.random.normal(mean=4, stddev=0.1, shape=x.shape)

            def squared_loss():
                y_pred = x * param1  # this is very simple, but we can of course use any
                                     # zfit PDF or Func inside
                squared = (y_pred - y) ** 2
                mse = tf.reduce_mean(squared)
                return mse

            loss = zfit.loss.SimpleLoss(squared_loss, param1)

        which can then be used in conjunction with any zfit minimizer such as Minuit

        .. code:: python

            minimizer = zfit.minize.Minuit()
            result = minimizer.minimize(loss)
        """
        super().__init__(model=[], data=[], options={'subtr_const': False})
        if dependents is not NONE and params is None:
            params = dependents
        elif deps is not NONE and params is None:  # depreceation
            params = deps
        elif params is None:  # legacy, remove in 0.7
            raise BreakingAPIChangeError("params need to be specified explicitly due to the upgrade to 0.4."
                                         "More information can be found in the upgrade guide on the website.")

        # @z.function(wraps='loss')
        # def wrapped_func():
        #     return func()
        if hasattr(func, 'errordef'):
            errordef = func.errordef

        if errordef is None:
            raise ValueError(f"{self} cannot minimize {func} as `errordef` is missing: "
                             f"it has to be set as an attribute. Typically 1 (chi2) or 0.5 (NLL).")

        sig = inspect.signature(func)
        self._call_with_args = len(sig.parameters) > 0

        self._simple_func = func
        self._errordef = errordef
        params = convert_to_parameters(params, prefer_constant=False)
        self._simple_func_params = _extract_dependencies(params)

    def _get_dependencies(self):
        dependents = self._simple_func_params
        return dependents

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = super()._get_params(floating, is_yield, extract_independent)
        params = params.union(self._simple_func_params)
        return params

    @property
    def errordef(self):
        errordef = self._errordef
        if errordef is None:
            raise RuntimeError("For this SimpleLoss, no error calculation is possible.")
        else:
            return errordef

    # @z.function(wraps='loss')
    def _loss_func(self, model, data, fit_range, constraints=None, log_offset=None):

        if self._call_with_args:
            params = self._simple_func_params

            value = self._simple_func(params)
        else:
            value = self._simple_func()
        return z.convert_to_tensor(value)

    def __add__(self, other):
        raise IntentionAmbiguousError("Cannot add a SimpleLoss, 'addition' of losses can mean anything."
                                      "Add them manually")

    def _cache_add_constraints(self, constraints):
        raise WorkInProgressError("Needed? will probably provided in future")
