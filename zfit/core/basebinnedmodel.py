#  Copyright (c) 2019 zfit
import contextlib
from typing import Union, Dict, Callable

import numpy as np
import tensorflow as tf

from zfit import ztf
from ..util.exception import BasePDFSubclassingError, ShapeIncompatibleError, NormRangeNotImplementedError, \
    MultipleLimitsNotImplementedError
from ..settings import ztypes
from ..util import ztyping
from ..util import container as zcontainer
from . import integration as zintegrate
from .interfaces import ZfitBinnedModel, ZfitParameter, ZfitHistData, ZfitBinningScheme
from .dimension import BaseDimensional
from ..util.cache import Cachable
from .limits import Space, convert_to_space, supports
from .baseobject import BaseNumeric

_BaseModel_USER_IMPL_METHODS_TO_CHECK = {}


def _BaseModel_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support (bool): If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.

    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func (function):

        Returns:
            function:
        """
        name = func.__name__
        _BaseModel_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BaseModel_register_check_support
        return func

    return register


class BinningScheme(ZfitBinningScheme):
    def __init__(self, edges):
        self._edges = edges

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    def compatible(self, binning: ZfitBinningScheme) -> bool:
        if not isinstance(binning, ZfitBinningScheme):
            raise TypeError("Cannot only compare with another ZfitBinningScheme")
        return np.allclose(self.edges, binning.edges)


class BaseBinnedModel(BaseNumeric, Cachable, BaseDimensional, ZfitBinnedModel):
    _DEFAULTS_integration = zcontainer.DotDict()
    _DEFAULTS_integration.mc_sampler = lambda *args, **kwargs: mc.sample_halton_sequence(*args, randomized=False,
                                                                                         **kwargs)
    # _DEFAULTS_integration.mc_sampler = lambda dim, num_results, dtype: tf.random_uniform(maxval=1.,
    #                                                                                      shape=(num_results, dim),
    #                                                                                      dtype=dtype)
    _DEFAULTS_integration.draws_per_dim = 40000
    _DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(self, obs: ztyping.ObsTypeInput, params: Union[Dict[str, ZfitParameter], None] = None,
                 binning: BinningScheme = None,
                 name: str = "BaseModel", dtype=ztypes.float,
                 **kwargs):
        """The base model to inherit from and overwrite `_unnormalized_pdf`.

        Args:
            dtype (DType): the dtype of the model
            name (str): the name of the model
            params (Dict(str, :py:class:`~zfit.Parameter`)): A dictionary with the internal name of the parameter and
                the parameters itself the model depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, obs=obs, **kwargs)
        self._binning = binning

        self._integration = zcontainer.DotDict()
        self._integration.auto_numeric_integrator = self._DEFAULTS_integration.auto_numeric_integrator
        self.integration = zintegrate.Integration(mc_sampler=self._DEFAULTS_integration.mc_sampler,
                                                  draws_per_dim=self._DEFAULTS_integration.draws_per_dim)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if subclass has decorator if required
        cls._subclass_check_support(methods_to_check=_BaseModel_USER_IMPL_METHODS_TO_CHECK,
                                    wrapper_not_overwritten=_BaseModel_register_check_support)
        cls._analytic_integral = zintegrate.AnalyticIntegral()
        cls._inverse_analytic_integral = []
        cls._additional_repr = {}

    @classmethod
    def _subclass_check_support(cls, methods_to_check, wrapper_not_overwritten):
        for method_name, has_support in methods_to_check.items():
            method = getattr(cls, method_name)
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == wrapper_not_overwritten:
                    continue  # not overwritten, fine

            # here means: overwritten
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == supports:
                    if has_support:
                        continue  # needs support, has been wrapped
                    else:
                        raise BasePDFSubclassingError("Method {} has been wrapped with supports "
                                                      "but is not allowed to. Has to handle all "
                                                      "arguments.".format(method_name))
                elif has_support:
                    raise BasePDFSubclassingError("Method {} has been overwritten and *has to* be "
                                                  "wrapped by `supports` decorator (don't forget () )"
                                                  "to call the decorator as it takes arguments"
                                                  "".format(method_name))
                elif not has_support:
                    continue  # no support, has not been wrapped with
            else:
                if not has_support:
                    continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            raise BasePDFSubclassingError("Method {} has not been correctly wrapped with @supports "
                                          "OR has been wrapped but it should not be".format(method_name))

    @contextlib.contextmanager
    def _convert_sort_x(self, x: ztyping.XTypeInput) -> ZfitHistData:
        if not isinstance(x, ZfitHistData):
            raise TypeError(f"x is not a ZfitHistData but {type(x)}")
        if x.obs is not None:
            with x.sort_by_obs(obs=self.obs, allow_superset=True):
                yield x
        elif x.axes is not None:
            with x.sort_by_axes(axes=self.axes):
                yield x
        else:
            assert False, "Neither the `obs` nor the `axes` are specified in `Data`"

    def _check_input_norm_range(self, norm_range, caller_name="",
                                none_is_error=False) -> Union[Space, bool]:
        """Convert to :py:class:`~zfit.Space`.

        Args:
            norm_range (None or :py:class:`~zfit.Space` compatible):
            caller_name (str): name of the calling function. Used for exception message.
            none_is_error (bool): if both `norm_range` and `self.norm_range` are None, the default
                value is `False` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two `None` will raise a Value error.

        Returns:
            Union[:py:class:`~zfit.Space`, False]:

        """
        if norm_range is None or (isinstance(norm_range, Space) and norm_range.limits is None):
            if none_is_error:
                raise ValueError("Normalization range `norm_range` has to be specified when calling {name} or"
                                 "a default normalization range has to be set. Currently, both are None"
                                 "".format(name=caller_name))
            # else:
            #     norm_range = False
        # if norm_range is False and not convert_false:
        #     return False

        return self.convert_sort_space(limits=norm_range)

    def _check_input_limits(self, limits, caller_name="", none_is_error=False):
        if limits is None or (isinstance(limits, Space) and limits.limits is None):
            if none_is_error:
                raise ValueError("The `limits` have to be specified when calling {name} and not be None"
                                 "".format(name=caller_name))
            # else:
            #     limits = False

        return self.convert_sort_space(limits=limits)

    def convert_sort_space(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                           limits: ztyping.LimitsTypeInput = None) -> Union[Space, None]:
        """Convert the inputs (using eventually `obs`, `axes`) to :py:class:`~zfit.Space` and sort them according to
        own `obs`.

        Args:
            obs ():
            axes ():
            limits ():

        Returns:

        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        self_space = self.space
        if self_space is not None:
            space = space.with_obs_axes(self_space.get_obs_axes(), ordered=True, allow_subset=True)
        return space

    @property
    def binning(self):
        return self._binning

    # TODO from here
    @_BaseModel_register_check_support(True)
    def _integrate(self, limits, norm_range):
        raise NotImplementedError()

    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, :py:class:`~zfit.Space`): the limits to integrate over
            norm_range (tuple, :py:class:`~zfit.Space`): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str): name of the operation shown in the :py:class:`tf.Graph`

        Returns:
            :py:class`tf.Tensor`: the integral value as a scalar with shape ()
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        limits = self._check_input_limits(limits=limits)
        integral = self._single_hook_integrate(limits=limits, norm_range=norm_range, name=name)
        if isinstance(integral, tf.Tensor):
            if not integral.shape.as_list() == []:
                raise ShapeIncompatibleError("Error in integral creation, should return an integral "
                                             "with shape () (resp. [] as list), current shape "
                                             "{}. If you registered an analytic integral which is used"
                                             "now, make sure to return a scalar and not a tensor "
                                             "(typically: shape is (1,) insead of () -> return tensor[0] "
                                             "instead of tensor)".format(integral.shape.as_list()))
        return integral

    def _single_hook_integrate(self, limits, norm_range, name):
        return self._hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_integrate(self, limits, norm_range, name='hook_integrate'):
        return self._norm_integrate(limits=limits, norm_range=norm_range, name=name)

    def _norm_integrate(self, limits, norm_range, name='norm_integrate'):
        try:
            integral = self._limits_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            unnormalized_integral = self._limits_integrate(limits=limits, norm_range=False, name=name)
            normalization = self._limits_integrate(limits=norm_range, norm_range=False, name=name)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.iter_limits(as_tuple=False):
                integrals.append(self._call_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(tf.stack(integrals), axis=0)
        return integral

    def _call_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with contextlib.suppress(NotImplementedError):
                return self._integrate(limits=limits, norm_range=norm_range)
            with contextlib.suppress(NotImplementedError):
                return self._hook_analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_integrate(limits=limits, norm_range=norm_range)

    def _fallback_integrate(self, limits, norm_range):
        axes = limits.axes
        max_axes = self._analytic_integral.get_max_axes(limits=limits, axes=axes)

        integral = None
        if max_axes and integral:  # TODO improve handling of available analytic integrals
            with contextlib.suppress(NotImplementedError):
                def part_int(x):
                    """Temporary partial integration function."""
                    return self._hook_partial_analytic_integrate(x, limits=limits, norm_range=norm_range)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._hook_numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    @classmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None,
                                   priority: Union[int, float] = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral with the class.

        Args:
            func (callable): A function that calculates the (partial) integral over the axes `limits`.
                The signature has to be the following:

                    * x (:py:class:`~zfit.core.interfaces.ZfitData`, None): the data for the remaining axes in a partial
                        integral. If it is not a partial integral, this will be None.
                    * limits (:py:class:`~zfit.Space`): the limits to integrate over.
                    * norm_range (:py:class:`~zfit.Space`, None): Normalization range of the integral.
                        If not `supports_supports_norm_range`, this will be None.
                    * params (Dict[param_name, :py:class:`zfit.Parameters`]): The parameters of the model.
                    * model (:py:class:`~zfit.core.interfaces.ZfitModel`):The model that is being integrated.

            limits (): |limits_arg_descr|
            priority (int): Priority of the function. If multiple functions cover the same space, the one with the
                highest priority will be used.
            supports_multiple_limits (bool): If `True`, the `limits` given to the integration function can have
                multiple limits. If `False`, only simple limits will pass through and multiple limits will be
                auto-handled.
            supports_norm_range (bool): If `True`, `norm_range` argument to the function may not be `None`.
                If `False`, `norm_range` will always be `None` and care is taken of the normalization automatically.

        """
        cls._analytic_integral.register(func=func, limits=limits, supports_norm_range=supports_norm_range,
                                        priority=priority, supports_multiple_limits=supports_multiple_limits)

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""

        # with tf.name_scope(self.name):
        with tf.compat.v1.name_scope(name, values=([] if values is None else values)) as scope:
            yield scope


if __name__ == '__main__':
    import zfit

    obs = zfit.Space('asdf', (-3, 5))
    binned_model = BaseBinnedModel()
