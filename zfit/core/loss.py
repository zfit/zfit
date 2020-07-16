#  Copyright (c) 2020 zfit

import abc
import warnings
from typing import Optional, Union, List, Callable, Iterable, Tuple, Set

import tensorflow as tf
from ordered_set import OrderedSet

from .baseobject import BaseNumeric
from .constraint import BaseConstraint
from .dependents import _extract_dependencies
from .interfaces import ZfitLoss, ZfitSpace, ZfitModel, ZfitData
from .. import z, settings
from ..util import ztyping
from ..util.checks import NOT_SPECIFIED
from ..util.container import convert_to_container, is_container
from ..util.exception import IntentionAmbiguousError, NotExtendedPDFError, WorkInProgressError, \
    BreakingAPIChangeError
from ..util.warnings import warn_advanced_feature
from ..z.math import numerical_gradient, autodiff_gradient, autodiff_value_gradients, numerical_value_gradients, \
    automatic_value_gradients_hessian, numerical_value_gradients_hessian


# @z.function
def _unbinned_nll_tf(model: ztyping.PDFInputType, data: ztyping.DataInputType, fit_range: ZfitSpace):
    """Return unbinned negative log likelihood graph for a PDF

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
        nlls = [_unbinned_nll_tf(model=p, data=d, fit_range=r)
                for p, d, r in zip(model, data, fit_range)]
        nll_finished = tf.reduce_sum(input_tensor=nlls, axis=0)
    else:
        with data.set_data_range(fit_range):
            probs = model.pdf(data, norm_range=fit_range)
        log_probs = tf.math.log(probs)
        nll = _nll_calc_unbinned_tf(log_probs=log_probs,
                                    weights=data.weights if data.weights is not None else None)
        nll_finished = nll
    return nll_finished


@z.function
def _nll_calc_unbinned_tf(log_probs, weights=None, log_offset=None):
    if weights is not None:
        log_probs *= weights  # because it's prob ** weights
    if log_offset:
        log_probs -= log_offset
    nll = -tf.reduce_sum(input_tensor=log_probs, axis=0)
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
                 constraints: ztyping.ConstraintsTypeInput = None):
        # first doc line left blank on purpose, subclass adds class docstring (Sphinx autodoc adds the two)
        """

        A "simultaneous fit" can be performed by giving one or more `model`, `data`, `fit_range`
        to the loss. The length of each has to match the length of the others.

        Args:
            model: The model or models to evaluate the data on
            data: Data to use
            fit_range: The fitting range. It's the norm_range for the models (if
            they
                have a norm_range) and the data_range for the data.
            constraints: A Tensor representing a loss constraint. Using
                `zfit.constraint.*` allows for easy use of predefined constraints.
        """
        super().__init__(name=type(self).__name__, params={})
        if fit_range is not None:
            warnings.warn("The fit_range argument is depreceated and will maybe removed in future releases. "
                          "It is preferred to define the range in the space"
                          " when creating the data and the model.")

        self.computed_gradients = {}
        model, data, fit_range = self._input_check(pdf=model, data=data, fit_range=fit_range)
        self._model = model
        self._data = data
        self._fit_range = fit_range
        if constraints is None:
            constraints = []
        self._constraints = _constraint_check_convert(convert_to_container(constraints, list))

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
        if not is_container(pdf):
            if isinstance(fit_range, list):
                raise TypeError("`pdf` and `data` are not a `list`, `fit_range` can't be a `list` then.")
        if isinstance(pdf, tuple):
            raise TypeError("`pdf` has to be a pdf or a list of pdfs, not a tuple.")

        if isinstance(data, tuple):
            raise TypeError("`data` has to be a data or a list of data, not a tuple.")

        pdf, data = (convert_to_container(obj, non_containers=[tuple]) for obj in (pdf, data))
        # TODO: data, range consistency?
        if fit_range is None:
            fit_range = []
            for p, d in zip(pdf, data):
                non_consistent = {'data': [], 'model': [], 'range': []}
                if not p.norm_range == d.data_range:
                    non_consistent['data'].append(d)
                    non_consistent['model'].append(p)
                    non_consistent['range'].append((p.norm_range, d.data_range))
                fit_range.append(p.norm_range)
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
        fit_range = [p.convert_sort_space(limits=range_) for p, range_ in zip(pdf, fit_range)]
        # TODO: sanitize pdf, data?
        self.add_cache_deps(cache_deps=pdf)
        self.add_cache_deps(cache_deps=data)
        self.add_cache_deps(cache_deps=fit_range)
        return pdf, data, fit_range

    def _input_check_params(self, params):
        if params is None:
            params = list(self.get_params())
        else:
            params = convert_to_container(params)
        return params

    def gradients(self, params: ztyping.ParamTypeInput = None) -> List[tf.Tensor]:
        params = self._input_check_params(params)
        return self._gradients(params=params)

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
        fit_range = self._fit_range
        return fit_range

    @property
    def constraints(self):
        return self._constraints

    def _get_dependencies(self):  # TODO: fix, add constraints
        pdf_dependents = _extract_dependencies(self.model)
        pdf_dependents |= _extract_dependencies(self.constraints)
        return pdf_dependents

    @abc.abstractmethod
    def _loss_func(self, model, data, fit_range, constraints):
        raise NotImplementedError

    def value(self):
        return self._value()

    @property
    def errordef(self) -> Union[float, int]:
        return self._errordef

    def _value(self):
        try:
            return self._loss_func(model=self.model, data=self.data, fit_range=self.fit_range,
                                   constraints=self.constraints)
        except NotImplementedError as error:
            raise NotImplementedError("_loss_func not properly defined!") from error

    def __add__(self, other):
        if not isinstance(other, BaseLoss):
            raise TypeError("Has to be a subclass of `BaseLoss` or overwrite `__add__`.")
        if not type(other) == type(self):
            raise ValueError("cannot safely add two different kind of loss.")
        model = self.model + other.model
        data = self.data + other.data
        fit_range = self.fit_range + other.fit_range
        constraints = self.constraints + other.constraints
        loss = type(self)(model=model, data=data, fit_range=fit_range, constraints=constraints)
        return loss

    def _gradients(self, params):
        if settings.options['numerical_grad']:
            gradients = numerical_gradient(self.value, params=params)

        else:
            gradients = autodiff_gradient(self.value, params=params)
        return gradients

    def value_gradients(self, params: ztyping.ParamTypeInput) -> Tuple[tf.Tensor, tf.Tensor]:
        params = self._input_check_params(params)
        return self._value_gradients(params=params)

    def _value_gradients(self, params):
        if settings.options['numerical_grad']:
            value, gradients = numerical_value_gradients(self.value, params=params)
        else:
            value, gradients = autodiff_value_gradients(self.value, params=params)
        return value, gradients

    def value_gradients_hessian(self, params: ztyping.ParamTypeInput, hessian=None) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
        params = self._input_check_params(params)
        numerical = settings.options['numerical_grad']
        vals = self._value_gradients_hessian(params=params, hessian=hessian, numerical=numerical)

        return vals

    @z.function(wraps='loss')
    def _value_gradients_hessian(self, params, hessian, numerical=False):
        if numerical:
            result = numerical_value_gradients_hessian(self.value, params=params, hessian=hessian)
        else:
            result = automatic_value_gradients_hessian(self.value, params=params, hessian=hessian)
        return result

    def __repr__(self) -> str:
        class_name = repr(self.__class__)[:-2].split(".")[-1]
        string = f'<{class_name} ' \
                 f'model={one_two_many([model.name for model in self.model])} ' \
                 f'data={one_two_many([data.name for data in self.data])} ' \
                 f'constraints={one_two_many(self.constraints, many="True")} ' \
                 f'>'
        return string

    def __str__(self) -> str:
        class_name = repr(self.__class__)[:-2].split(".")[-1]
        string = f'<{class_name}' \
                 f' model={one_two_many([model for model in self.model])}' \
                 f' data={one_two_many([data for data in self.data])}' \
                 f' constraints={one_two_many(self.constraints, many="True")}' \
                 f'>'
        return string


def one_two_many(values, n=3, many='multiple'):
    values = convert_to_container(values)
    if len(values) > n:
        values = many
    return values


class CachedLoss(BaseLoss):

    def __init__(self, model, data, fit_range=None, constraints=None):
        raise WorkInProgressError("Currently, caching is not implemented in the loss and does not make"
                                  "sense, it is 'not yet upgraded to TF2'")
        super().__init__(model=model, data=data, fit_range=fit_range, constraints=constraints)

    @abc.abstractmethod
    def _cache_add_constraints(self, constraints):
        raise NotImplementedError

    def _value(self):
        if self._cache.get('loss') is None:
            loss = super()._value()
            self._cache['loss'] = loss
        else:
            loss = self._cache['loss']
        return loss

    def _add_constraints(self, constraints):
        super()._add_constraints(constraints=constraints)
        self._cache_add_constraints(constraints=constraints)

    def _gradients(self, params):
        params_cache = self._cache.get('gradients', {})
        params_todo = []
        for param in params:
            if param not in params_cache:
                params_todo.append(param)
        if params_todo:
            gradients = {(p, grad) for p, grad in zip(params_todo, super()._gradients(params_todo))}
            params_cache.update(gradients)

        self._cache['gradients'] = params_cache

        param_gradients = [params_cache[param] for param in params]
        return param_gradients


# class UnbinnedNLL(CachedLoss):
class UnbinnedNLL(BaseLoss):
    """The Unbinned Negative Log Likelihood."""

    _name = "UnbinnedNLL"

    def __init__(self, model, data, fit_range=None, constraints=None):
        super().__init__(model=model, data=data, fit_range=fit_range, constraints=constraints)
        self._errordef = 0.5
        extended_pdfs = [pdf for pdf in self.model if pdf.is_extended]
        if extended_pdfs and type(self) == UnbinnedNLL:
            warn_advanced_feature("Extended PDFs are given to a normal UnbinnedNLL. This won't take the yield "
                                  "into account and simply treat the PDFs as non-extended PDFs. To create an "
                                  "extended NLL, use the `ExtendedUnbinnedNLL`.", identifier='extended_in_UnbinnedNLL')

    @z.function(wraps='loss')
    def _loss_func(self, model, data, fit_range, constraints):
        nll = self._loss_func_watched(constraints, data, fit_range, model)

        return nll

    @property
    def is_extended(self):
        return False

    @z.function(wraps='loss')
    def _loss_func_watched(self, constraints, data, fit_range, model):
        nll = _unbinned_nll_tf(model=model, data=data, fit_range=fit_range)
        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            nll += constraints
        return nll

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        if not self.is_extended:
            is_yield = False  # the loss does not depend on the yields
        return super()._get_params(floating, is_yield, extract_independent)

    # def _cache_add_constraints(self, constraints):
    #     if self._cache.get('loss') is not None:
    #         constraints = [c.value() for c in constraints]
    #         self._cache['loss'] += z.reduce_sum(constraints)


class ExtendedUnbinnedNLL(UnbinnedNLL):
    """An Unbinned Negative Log Likelihood with an additional poisson term for the"""

    @z.function(wraps='loss')
    def _loss_func(self, model, data, fit_range, constraints):
        nll = super()._loss_func(model=model, data=data, fit_range=fit_range, constraints=constraints)
        yields = []
        nevents_collected = []
        for mod, dat in zip(model, data):
            if not mod.is_extended:
                raise NotExtendedPDFError("The pdf {} is not extended but has to be (for an extended fit)".format(mod))
            nevents = dat.n_events if dat.weights is None else z.reduce_sum(dat.weights)
            nevents = tf.cast(nevents, tf.float64)
            nevents_collected.append(nevents)
            yields.append(mod.get_yield())
        yields = tf.stack(yields, axis=0)
        nevents_collected = tf.stack(nevents_collected, axis=0)

        term_new = tf.nn.log_poisson_loss(nevents_collected, tf.math.log(yields))
        nll += tf.reduce_sum(term_new, axis=0)
        return nll

    @property
    def is_extended(self):
        return True


class SimpleLoss(BaseLoss):
    _name = "SimpleLoss"

    def __init__(self, func: Callable, deps: Iterable["zfit.Parameter"] = NOT_SPECIFIED,
                 dependents: Iterable["zfit.Parameter"] = NOT_SPECIFIED,
                 errordef: Optional[float] = None):
        """Loss from a (function returning a) Tensor.

        Args:
            func: Callable that constructs the loss and returns a tensor.
            deps: The dependents (independent `zfit.Parameter`) of the loss. If not given, the dependents are
                figured out automatically.
            errordef: Definition of which change in the loss corresponds to a change of 1 sigma.
                For example, 1 for Chi squared, 0.5 for negative log-likelihood.
        """
        if dependents is not NOT_SPECIFIED:
            warnings.warn("`dependents` is deprecated and will be removed in the future, use `deps`"
                          " instead as a keyword.")
        if deps is NOT_SPECIFIED:  # depreceation
            raise BreakingAPIChangeError("Dependents need to be specified explicitly due to the upgrade to 0.4."
                                         "More information can be found in the upgrade guide on the website.")

        @z.function(wraps='loss')
        def wrapped_func():
            return func()

        self._simple_func = wrapped_func
        self._simple_errordef = errordef
        self._errordef = errordef
        self.computed_gradients = {}
        deps = convert_to_container(deps, container=OrderedSet)
        self._simple_func_deps = _extract_dependencies(deps)

        super().__init__(model=[], data=[], fit_range=[])

    def _get_dependencies(self):
        dependents = self._simple_func_deps
        return dependents

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = super()._get_params(floating, is_yield, extract_independent)
        params = params.union(self._simple_func_deps)
        return params

    @property
    def errordef(self):
        errordef = self._simple_errordef
        if errordef is None:
            errordef = -999
            # raise RuntimeError("For this SimpleLoss, no error calculation is possible.")
        else:
            return errordef

    def _loss_func(self, model, data, fit_range, constraints=None):
        return self._simple_func()

    def __add__(self, other):
        raise IntentionAmbiguousError("Cannot add a SimpleLoss, 'addition' of losses can mean anything."
                                      "Add them manually")

    def _cache_add_constraints(self, constraints):
        raise WorkInProgressError("Needed? will probably provided in future")
