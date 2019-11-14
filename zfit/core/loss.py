#  Copyright (c) 2019 zfit

import abc
from collections import OrderedDict

import numdifftools
import tensorflow as tf
import numpy as np

from typing import Optional, Union, List, Callable, Iterable

from zfit import ztf, settings
from zfit.util import ztyping
from zfit.util.cache import Cachable
from zfit.util.checks import ZfitNotImplemented
from zfit.util.graph import get_dependents_auto
from .baseobject import BaseObject
from zfit.core.dependents import BaseDependentsMixin
from .interfaces import ZfitLoss, ZfitSpace, ZfitModel, ZfitData, ZfitPDF
from ..models.functions import SimpleFunc
from ..util.container import convert_to_container, is_container
from ..util.exception import IntentionNotUnambiguousError, NotExtendedPDFError, DueToLazynessNotImplementedError
from zfit.settings import ztypes
from .constraint import BaseConstraint, SimpleConstraint

func_simple = tf.function(autograph=False)  # TODO: how to properly?


# @func_simple
def _unbinned_nll_tf(model: ztyping.PDFInputType, data: ztyping.DataInputType, fit_range: ZfitSpace):
    """Return unbinned negative log likelihood graph for a PDF

    Args:
        model (ZfitModel): PDFs with a `.pdf` method. Has to be as many models as data
        data (ZfitData):
        fit_range ():

    Returns:
        graph: the unbinned nll

    Raises:
        ValueError: if both `probs` and `log_probs` are specified.
    """

    if is_container(model):
        nlls = [_unbinned_nll_tf(model=p, data=d, fit_range=r)
                for p, d, r in zip(model, data, fit_range)]
        nll_finished = tf.reduce_sum(input_tensor=nlls)
    else:
        with data.set_data_range(fit_range):
            probs = model.pdf(data, norm_range=fit_range)
        log_probs = tf.math.log(probs)
        if data.weights is not None:
            log_probs *= data.weights  # because it's prob ** weights
        nll = -tf.reduce_sum(input_tensor=log_probs)
        nll_finished = nll
    return nll_finished


def _nll_constraints_tf(constraints):
    if not constraints:
        return ztf.constant(0.)  # adding 0 to nll
    probs = []
    for param, dist in constraints.items():
        probs.append(dist.pdf(param))
    # probs = [dist.pdf(param) for param, dist in constraints.items()]
    constraints_neg_log_prob = -tf.reduce_sum(input_tensor=tf.math.log(probs))
    return constraints_neg_log_prob


def _constraint_check_convert(constraints):
    checked_constraints = []
    for constr in constraints:
        if isinstance(constr, BaseConstraint):
            checked_constraints.append(constr)
        else:
            checked_constraints.append(SimpleConstraint(func=lambda: constr))
    return checked_constraints


class BaseLoss(BaseDependentsMixin, ZfitLoss, Cachable, BaseObject):

    def __init__(self, model, data, fit_range: ztyping.LimitsTypeInput = None, constraints: List[tf.Tensor] = None):
        # first doc line left blank on purpose, subclass adds class docstring (Sphinx autodoc adds the two)
        """

        A "simultaneous fit" can be performed by giving one or more `model`, `data`, `fit_range`
        to the loss. The length of each has to match the length of the others.

        Args:
            model (Iterable[ZfitModel]): The model or models to evaluate the data on
            data (Iterable[ZfitData]): Data to use
            fit_range (Iterable[:py:class:`~zfit.Space`]): The fitting range. It's the norm_range for the models (if
            they
                have a norm_range) and the data_range for the data.
            constraints (Iterable[tf.Tensor): A Tensor representing a loss constraint. Using
                `zfit.constraint.*` allows for easy use of predefined constraints.
        """
        super().__init__(name=type(self).__name__)
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
                if not p.norm_range == d.data_range:
                    raise IntentionNotUnambiguousError("No `fit_range` is specified and `pdf` {} as "
                                                       "well as `data` {} have different ranges they"
                                                       "are defined in. Either make them (all) consistent"
                                                       "or specify the `fit_range`")
                fit_range.append(p.norm_range)
        else:
            fit_range = convert_to_container(fit_range, non_containers=[tuple])

        # simultaneous fit
        # if is_container(pdf):
        # if not is_container(fit_range) or not isinstance(fit_range[0], Space):
        #     raise ValueError(
        #         "If several pdfs are specified, the `fit_range` has to be given as a list of `Space` "
        #         "objects and not as pure tuples.")

        # else:
        #     fit_range = pdf.convert_sort_space(limits=fit_range)  # fit_range may be a tuple
        if not len(pdf) == len(data) == len(fit_range):
            raise ValueError("pdf, data and fit_range don't have the same number of components:"
                             "\npdf: {}"
                             "\ndata: {}"
                             "\nfit_range: {}".format(pdf, data, fit_range))

        # sanitize fit_range
        fit_range = [p.convert_sort_space(limits=range_) for p, range_ in zip(pdf, fit_range)]
        # TODO: sanitize pdf, data?
        self.add_cache_dependents(cache_dependents=pdf)
        self.add_cache_dependents(cache_dependents=data)
        self.add_cache_dependents(cache_dependents=fit_range)
        return pdf, data, fit_range

    def gradients(self, params: ztyping.ParamTypeInput = None) -> List[tf.Tensor]:
        if params is None:
            params = list(self.get_dependents())
        else:
            params = convert_to_container(params)
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

    def _get_dependents(self):
        pdf_dependents = self._extract_dependents(self.model)
        return pdf_dependents

    @abc.abstractmethod
    def _loss_func(self, model, data, fit_range, constraints):
        raise NotImplementedError

    # @func_simple
    def value(self):
        return self._value()

    @property
    def errordef(self) -> Union[float, int]:
        return self._errordef

    def _value(self):
        try:
            return self._loss_func(model=self.model, data=self.data, fit_range=self.fit_range,
                                   constraints=self.constraints)
        except NotImplementedError:
            raise NotImplementedError("_loss_func not properly defined!")

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

    # def _gradients(self, params):
    #     tf.gradients(ys=self.value(), xs=params)
    #     return [self.computed_gradients[param] for param in params]
    def _gradients(self, params):
        if settings.options['numerical_grad']:
            def loss_func(param_values):
                for param, value in zip(params, param_values):
                    param.assign(value)
                # return run(self.value())
                return self.value().numpy()

            param_vals = tf.stack(params)
            original_vals = param_vals.numpy()
            grad_func = numdifftools.Gradient(loss_func)
            gradients = tf.py_function(grad_func, inp=[param_vals],
                                       Tout=tf.float64)
            gradients.set_shape((len(param_vals),))
            for param, val in zip(params, original_vals):
                param.set_value(val)
            # gradients = grad_func(param_vals)

        else:
            with tf.GradientTape(persistent=False) as tape:
                loss = self.value()

            # variables = tape.watched_variables()
            gradients = tape.gradient(loss, sources=params)
            # gradients = [self.computed_gradients[param] for param in params]
        return gradients

    def value_gradients(self, params):
        return self._value_gradients(params=params)

    def _value_gradients(self, params):

        with tf.GradientTape(persistent=False) as tape:
            loss = self.value()

        # variables = tape.watched_variables()
        gradients = tape.gradient(loss, sources=params)
        return loss, gradients

    def value_gradients_hessian(self, params):
        vals = self._value_gradients_hessian(params=params)
        if vals is ZfitNotImplemented:
            vals = self._value_gradients_hessian_fallback(params=params)
        return vals

    def _value_gradients_hessian_fallback(self, params):
        with tf.GradientTape(persistent=False) as tape:
            loss, gradients = self.value_gradients(params=params)

        # variables = tape.watched_variables()
        hesse = tape.gradient(gradients, sources=params)
        return loss, gradients, hesse

    def _value_gradients_hessian(self, params):
        return ZfitNotImplemented


class CachedLoss(BaseLoss):

    def __init__(self, model, data, fit_range=None, constraints=None):
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

    def _loss_func(self, model, data, fit_range, constraints):
        with tf.GradientTape(persistent=True) as tape:
            nll = self._loss_func_watched(constraints, data, fit_range, model)

        variables = tape.watched_variables()
        gradients = tape.gradient(nll, sources=variables)
        for param, grad in zip(variables, gradients):
            # if param in self.computed_gradients:
            #     continue
            self.computed_gradients[param] = grad
        return nll

    @tf.function(autograph=False)
    def _loss_func_watched(self, constraints, data, fit_range, model):
        nll = _unbinned_nll_tf(model=model, data=data, fit_range=fit_range)
        if constraints:
            constraints = ztf.reduce_sum([c.value() for c in constraints])
            nll += constraints
        return nll

    def _cache_add_constraints(self, constraints):
        if self._cache.get('loss') is not None:
            constraints = [c.value() for c in constraints]
            self._cache['loss'] += ztf.reduce_sum(constraints)


class ExtendedUnbinnedNLL(UnbinnedNLL):
    """An Unbinned Negative Log Likelihood with an additional poisson term for the"""

    def _loss_func(self, model, data, fit_range, constraints):
        nll = super()._loss_func(model=model, data=data, fit_range=fit_range, constraints=constraints)
        poisson_terms = []
        for mod, dat in zip(model, data):
            if not mod.is_extended:
                raise NotExtendedPDFError("The pdf {} is not extended but has to be (for an extended fit)".format(mod))
            nevents = dat.nevents if dat.weights is None else ztf.reduce_sum(dat.weights)
            poisson_terms.append(-mod.get_yield() + ztf.to_real(nevents) * tf.math.log(mod.get_yield()))
        nll -= tf.reduce_sum(input_tensor=poisson_terms)
        return nll


class SimpleLoss(BaseLoss):
    _name = "SimpleLoss"

    def __init__(self, func: Callable, dependents: Optional[Iterable["zfit.Parameter"]] = None,
                 errordef: Optional[float] = None):
        """Loss from a (function returning a ) Tensor.

        Args:
            func: Callable that constructs the loss and returns a tensor.
            dependents: The dependents (independent `zfit.Parameter`) of the loss. If not given, the dependents are
                figured out automatically.
            errordef: Definition of which change in the loss corresponds to a change of 1 sigma.
                For example, 1 for Chi squared, 0.5 for negative log-likelihood.
        """
        # self._simple_func = tf.function(func, autograph=False)
        self._simple_func = tf.function(func, autograph=True)
        self._simple_errordef = errordef
        self._errordef = errordef
        self.computed_gradients = {}
        self._simple_func_dependents = convert_to_container(dependents, container=set)

        super().__init__(model=[], data=[], fit_range=[])

    def _get_dependents(self):
        dependents = self._simple_func_dependents
        if dependents is None:
            independent_params = tf.compat.v1.get_collection("zfit_independent")
            dependents = get_dependents_auto(tensor=self.value(), candidates=independent_params)
            self._simple_func_dependents = dependents
        return dependents

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
        raise IntentionNotUnambiguousError("Cannot add a SimpleLoss, 'addition' of losses can mean anything."
                                           "Add them manually")

    def _cache_add_constraints(self, constraints):
        raise DueToLazynessNotImplementedError("Needed? will probably provided in future")
