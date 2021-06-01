#  Copyright (c) 2021 zfit

import abc
from collections import OrderedDict
from typing import Callable, Dict, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
from ordered_set import OrderedSet

import zfit.z.numpy as znp
from zfit import z

from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import ShapeIncompatibleError
from .baseobject import BaseNumeric
from .dependents import _extract_dependencies
from .interfaces import ZfitConstraint, ZfitParameter
from .parameter import convert_to_parameter

tfd = tfp.distributions


class BaseConstraint(ZfitConstraint, BaseNumeric):

    def __init__(self, params: Union[Dict[str, ZfitParameter]] = None,
                 name: str = "BaseConstraint", dtype=ztypes.float,
                 **kwargs):
        """Base class for constraints.

        Args:
            dtype: the dtype of the constraint
            name: the name of the constraint
            params: A dictionary with the internal name of the
                parameter and the parameters itself the constrains depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, **kwargs)

    def value(self):
        return self._value()

    @abc.abstractmethod
    def _value(self):
        raise NotImplementedError

    def _get_dependencies(self) -> ztyping.DependentsType:
        return _extract_dependencies(self.get_params(floating=None))


class SimpleConstraint(BaseConstraint):

    def __init__(self, func: Callable, params: Optional[ztyping.ParameterType]):
        """Constraint from a (function returning a) Tensor.

        The parameters are named "param_{i}" with i starting from 0 and corresponding to the index of params.

        Args:
            func: Callable that constructs the constraint and returns a tensor.
            params: The dependents (independent `zfit.Parameter`) of the loss. If not given, the
                dependents are figured out automatically.
        """
        self._simple_func = func
        self._simple_func_dependents = convert_to_container(params, container=OrderedSet)

        params = convert_to_container(params, container=list)
        params = OrderedDict((f"param_{i}", p) for i, p in enumerate(params))

        super().__init__(name="SimpleConstraint", params=params)

    def _value(self):
        return self._simple_func()


class ProbabilityConstraint(BaseConstraint):

    def __init__(self, observation: Union[ztyping.NumericalScalarType, ZfitParameter],
                 params: Union[Dict[str, ZfitParameter]] = None, name: str = "ProbabilityConstraint",
                 dtype=ztypes.float, **kwargs):
        """Base class for constraints using a probability density function.

        Args:
            dtype: the dtype of the constraint
            name: the name of the constraint
            params: The parameters to constraint
            observation: Observed values of the parameter
                to constraint obtained from auxiliary measurements.
        """

        params_dict = {f"param_{i}": p for i, p in enumerate(params)}
        super().__init__(name=name, dtype=dtype, params=params_dict, **kwargs)

        observation = convert_to_container(observation, tuple)
        if len(observation) != len(params):
            raise ShapeIncompatibleError("observation and params have to be the same length. Currently"
                                         f"observation: {len(observation)}, params: {len(params)}")

        self._observation = []
        for obs, p in zip(observation, params):
            obs = convert_to_parameter(obs, f"{p.name}_obs", prefer_constant=False)
            obs.floating = False
            self._observation.append(obs)

        self._ordered_params = params

    @property
    def observation(self):
        """Return the observed values of the parameters constrained."""
        return self._observation

    def value(self):
        return self._value()

    @abc.abstractmethod
    def _value(self):
        raise NotImplementedError

    def _get_dependencies(self) -> ztyping.DependentsType:
        return _extract_dependencies(self.get_params())

    def sample(self, n):
        """Sample `n` points from the probability density function for the observed value of the parameters.

        Args:
            n: The number of samples to be generated.
        Returns:
        """
        sample = self._sample(n=n)
        return {p: sample[:, i] for i, p in enumerate(self.observation)}

    @abc.abstractmethod
    def _sample(self, n):
        raise NotImplementedError

    @property
    def _params_array(self):
        return z.convert_to_tensor(self._ordered_params)


class TFProbabilityConstraint(ProbabilityConstraint):

    def __init__(self, observation: Union[ztyping.NumericalScalarType, ZfitParameter],
                 params: Dict[str, ZfitParameter], distribution: tfd.Distribution,
                 dist_params, dist_kwargs=None, name: str = "DistributionConstraint", dtype=ztypes.float,
                 **kwargs):
        """Base class for constraints using a probability density function from `tensorflow_probability`.

        Args:
            distribution: The probability density function
                used to constraint the parameters
        """
        super().__init__(observation=observation, params=params, name=name, dtype=dtype, **kwargs)

        self._distribution = distribution
        self.dist_params = dist_params
        self.dist_kwargs = dist_kwargs if dist_kwargs is not None else {}

    @property
    def distribution(self):
        params = self.dist_params
        if callable(params):
            params = params(self.observation)
        kwargs = self.dist_kwargs
        if callable(kwargs):
            kwargs = kwargs()
        return self._distribution(**params, **kwargs, name=self.name + "_tfp")

    def _value(self):
        value = -self.distribution.log_prob(self._params_array)
        return value

    def _sample(self, n):
        # TODO cache: add proper caching
        return self.distribution.sample(n)


class GaussianConstraint(TFProbabilityConstraint):

    def __init__(self, params: ztyping.ParamTypeInput, observation: ztyping.NumericalScalarType,
                 uncertainty: ztyping.NumericalScalarType):
        r"""Gaussian constraints on a list of parameters to some observed values with uncertainties.

        A Gaussian constraint is defined as the likelihood of `params` given the `observations` and `uncertainty` from
        a different measurement.

        .. math::
            \text{constraint} = \text{Gauss}(\text{observation}; \text{params}, \text{uncertainty})


        Args:
            params: The parameters to constraint; corresponds to x in the Gaussian
                distribution.
            observation: observed values of the parameter; corresponds to mu
                in the Gaussian distribution.
            uncertainty: Uncertainties or covariance/error
                matrix of the observed values. Can either be a single value, a list of values, an array or a tensor.
                Corresponds to the sigma of the Gaussian distribution.
        Raises:
            ShapeIncompatibleError: If params, mu and sigma have incompatible shapes.
        """

        observation = convert_to_container(observation, tuple)
        params = convert_to_container(params, tuple)

        def create_covariance(mu, sigma):
            mu = z.convert_to_tensor(mu)
            sigma = z.convert_to_tensor(sigma)  # TODO (Mayou36): fix as above?
            params_tensor = z.convert_to_tensor(params)

            if sigma.shape.ndims > 1:
                covariance = sigma  # TODO: square as well?
            elif sigma.shape.ndims == 1:
                covariance = tf.linalg.tensor_diag(z.pow(sigma, 2.))
            else:
                sigma = znp.reshape(sigma, [1])
                covariance = tf.linalg.tensor_diag(z.pow(sigma, 2.))

            if not params_tensor.shape[0] == mu.shape[0] == covariance.shape[0] == covariance.shape[1]:
                raise ShapeIncompatibleError(f"params_tensor, observation and uncertainty have to have the"
                                             " same length. Currently"
                                             f"param: {params_tensor.shape[0]}, mu: {mu.shape[0]}, "
                                             f"covariance (from uncertainty): {covariance.shape[0:2]}")
            return covariance

        distribution = tfd.MultivariateNormalTriL
        dist_params = lambda observation: dict(loc=observation,
                                               scale_tril=tf.linalg.cholesky(
                                                   create_covariance(observation, uncertainty)))
        dist_kwargs = dict(validate_args=True)

        super().__init__(name="GaussianConstraint", observation=observation, params=params,
                         distribution=distribution, dist_params=dist_params, dist_kwargs=dist_kwargs)

        self._covariance = lambda: create_covariance(self.observation, uncertainty)

    @property
    def covariance(self):
        """Return the covariance matrix of the observed values of the parameters constrained."""
        return self._covariance()
