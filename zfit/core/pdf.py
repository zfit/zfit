#  Copyright (c) 2021 zfit
import typing
from contextlib import suppress
from typing import Callable

import tensorflow_probability as tfp
import zfit_interface.typing as ztyping
from zfit_interface.pdf import ZfitPDF
from zfit_interface.variables import ZfitVar, ZfitSpace, ZfitParam

from zfit import convert_to_parameter, z
from zfit._variables.varsupport import VarSupports
from zfit.core.func import Func
from zfit.core.values import ValueHolder
from zfit.util.container import convert_to_container
from zfit.util.exception import AlreadyExtendedPDFError, SpecificFunctionNotImplemented, NotExtendedPDFError


class Integration:
    _analytic_integrals = {}
    def __init__(self, mc_sampler=None, draws_per_dim=None, numeric_integrator=None):
        self._analytic_integrals = self._analytic_integrals.copy()
        if mc_sampler is None:
            mc_sampler = lambda *args, **kwargs: tfp.mcmc.sample_halton_sequence(*args, randomized=False,
                                                                                 **kwargs)
        if numeric_integrator is None:
            numeric_integrator = False #  TODO
        if draws_per_dim is None:
            draws_per_dim = 40_000
        self.numeric_integrator = numeric_integrator
        self.mc_sampler = mc_sampler
        self.draws_per_dim = draws_per_dim

    def register_on_object(self, var: ztyping.Variable, func: Callable, overwrite: bool = False):
        var = convert_to_container(var, frozenset)
        if var in self._analytic_integrals and not overwrite:
            raise ValueError(f"An analytic integral for {var} is already registered and 'overwrite' is "
                             f"set to False.")
        self._analytic_integrals[var] = func

    def get_available(self, var):
        var = convert_to_container(var, frozenset)
        candidates = sorted((v for v in self._analytic_integrals if var.issubset(v)), key=len)
        return {v: self._analytic_integrals[v] for v in candidates}

    @property
    def has_full(self, var):
        var = convert_to_container(var, frozenset)
        return len(list(self.get_available(var).keys()) + [[]][0]) == len(var)

    def has_partial(self, var):
        var = convert_to_container(var, frozenset)
        return bool(self.get_available(var))


class PDF(Func, ZfitPDF):

    def __init__(self, obs: typing.Mapping[str, ZfitSpace] = None, params: typing.Mapping[str, ZfitParam] = None,
                 var: typing.Mapping[str, ZfitVar] = None, extended: bool = None,
                 norm: typing.Mapping[str, ZfitSpace] = None,
                 label: typing.Optional[str] = None):
        if obs is not None:
            obs = {axis: VarSupports(var=ob.name, data=True)
                   for axis, ob in obs.items()
                   if not isinstance(ob, VarSupports)}
        else:
            obs = {}
        if params is None:
            params = {}
        else:
            params = {axis: VarSupports(var=p.name, scalar=True) for axis, p in params.items()}
        if var is None:
            var = {}
        else:
            var = var.copy()
            if not all(isinstance(v, VarSupports) for v in var.values()):
                raise TypeError(f"All var need to be VarSupports.")
        var.update(obs)
        var.update(params)
        super().__init__(var=var, label=label)
        if norm is None:
            norm = self.space
        self.norm = norm
        if extended is not None:
            self._set_yield(extended)

        self.integration = Integration()

    def _set_yield(self, value):
        # if self.is_extended:
        #     raise AlreadyExtendedPDFError(f"Cannot extend {self}, is already extended.")
        value = convert_to_parameter(value)
        # self.add_cache_deps(value)  # TODO
        self._yield = value

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the model is extended or not.

        Returns:
            A boolean.
        """
        return self._yield is not None

    def __call__(self, var):
        if self.is_extended:
            return self.ext_pdf(var)
        else:
            return self.pdf(var)

    def _pdf(self, var, norm):
        raise SpecificFunctionNotImplemented

    def pdf(self, var: ztyping.VarInputType, norm: ztyping.NormInputType = None, *,
            options=None) -> ztyping.PDFReturnType:
        """Probability density function, normalized over `norm`.

        Args:
          var: `float` or `double` `Tensor`.
          norm: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        var = self._convert_check_input_var(var)
        norm = self._convert_check_input_norm(norm, var=var)
        if var.space is not None:
            return self.integrate(limits=var, norm=norm, options=options)
        value = self._call_pdf(var=var, norm=norm, options=options)
        return value
        # with self._convert_sort_x(var) as var:
        #     value = self._single_hook_pdf(x=var, norm_range=norm)
        #     if run.numeric_checks:
        #         z.check_numerics(value, message="Check if pdf output contains any NaNs of Infs")
        #     return z.to_real(value)

    @z.function(wraps='model')
    def _call_pdf(self, var, norm, *, options=None):
        return self._pdf(var, norm)  # TODO

    def _ext_pdf(self, var, norm):
        raise SpecificFunctionNotImplemented

    def ext_pdf(self, var: ztyping.VarInputType, norm: ztyping.NormInputType = None, *,
            options=None) -> ztyping.PDFReturnType:
        """Probability density function, normalized over `norm`.

        Args:
          var: `float` or `double` `Tensor`.
          norm: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        if not self.is_extended:
            raise NotExtendedPDFError
        var = self._convert_check_input_var(var)
        norm = self._convert_check_input_norm(norm, var=var)
        if var.space is not None:
            return self.integrate(limits=var, norm=norm, options=options)
        return self._call_ext_pdf(var=var, norm=norm, options=options)

    @z.function(wraps='model')
    def _call_ext_pdf(self, var, norm, *, options=None):
        return self._ext_pdf(var, norm)  # TODO

    def _integrate(self,var, norm, options):
        raise SpecificFunctionNotImplemented

    def integrate(self, limits, norm=None, *, var=None, options=None):
        var = self._convert_check_input_var(limits, var)
        if var.space is None:
            raise ValueError(f"No space is given to integrate of {self}, needs at least one.")
        norm = self._convert_check_input_norm(norm, var=var)
        return self._call_integrate(var=var, norm=norm, options=options)

    @z.function(wraps='model')
    def _call_integrate(self, var, norm, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_integrate(var, norm, options=options)
        if self.is_extended:
            return self._auto_ext_integrate(var, norm, options=options) / self.get_yield()
        return self._fallback_integrate(var, norm, options=options)

    def _auto_integrate(self, var, norm, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._integrate(var, norm, options=options)
        return self._fallback_integrate(var=var, norm=norm, options=options)

    def _fallback_integrate(self, var, norm, options):
        pass

    def _ext_integrate(self, var, norm, options):
        raise SpecificFunctionNotImplemented

    def ext_integrate(self, limits, norm=None, *, var=None, options=None):
        if not self.is_extended:
            raise NotExtendedPDFError
        var = self._convert_check_input_var(limits, var)
        if var.space is None:
            raise ValueError(f"No space is given to integrate of {self}, needs at least one.")
        norm = self._convert_check_input_norm(norm, var=var)
        return self._call_ext_integrate(var=var, norm=norm, options=options)

    @z.function(wraps='model')
    def _call_ext_integrate(self, var, norm, options):
        with suppress(SpecificFunctionNotImplemented):
            return self._auto_ext_integrate(var, norm, options=options)
        if self.is_extended:
            return self._auto_integrate(var, norm, options=options) * self.get_yield()
        return self._fallback_ext_integrate(var, norm, options=options)

    def _auto_ext_integrate(self, var, norm, options):
        return self._ext_integrate(var, norm, options=options)

    def _fallback_ext_integrate(self, var, norm, options):
        pass  # TODO
        # return self.integration.mixed(var, norm, options)

    def _convert_check_input_var(self, var):
        var = ValueHolder(var)
        return var  # TODO

    def _convert_check_input_norm(self, norm, var):
        if norm is None:
            norm = self.norm
        # return var  # TODO


class HistPDF(PDF):

    def __init__(self, obs: typing.Mapping[str, ZfitSpace] = None, params: typing.Mapping[str, ZfitParam] = None,
                 var: typing.Mapping[str, ZfitVar] = None, extended: bool = None,
                 norm: typing.Mapping[str, ZfitSpace] = None, label: typing.Optional[str] = None):
        super().__init__(obs=obs, params=params, var=var, extended=extended, norm=norm, label=label)