#  Copyright (c) 2023 zfit

from __future__ import annotations

import abc
import collections
from collections import OrderedDict
from collections.abc import Callable
from typing import Mapping, Iterable, List

import numpy as np
import pydantic

from .serialmixin import SerializableMixin
from ..serialization.serializer import BaseRepr, Serializer

from typing import Literal

import tensorflow as tf
import tensorflow_probability as tfp
from ordered_set import OrderedSet

import zfit.z.numpy as znp
from zfit import z
from .baseobject import BaseNumeric
from .dependents import _extract_dependencies
from .interfaces import ZfitConstraint, ZfitParameter
from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import ShapeIncompatibleError

tfd = tfp.distributions


# TODO(serialization): add to serializer
class BaseConstraintRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["BaseConstraint"] = pydantic.Field("BaseConstraint", alias="type")


class BaseConstraint(ZfitConstraint, BaseNumeric):
    def __init__(
        self,
        params: dict[str, ZfitParameter] = None,
        name: str = "BaseConstraint",
        dtype=ztypes.float,
        **kwargs,
    ):
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


# TODO: improve arbitrary constraints, should we allow only functions that have a `params` argument?
class SimpleConstraint(BaseConstraint):
    def __init__(
        self,
        func: Callable,
        params: Mapping[str, ztyping.ParameterType]
        | Iterable[ztyping.ParameterType]
        | ztyping.ParameterType
        | None,
        *,
        name: str = None,
    ):
        """Constraint from a (function returning a) Tensor.

        Args:
            func: Callable that constructs the constraint and returns a tensor. For the expected signature,
                see below in ``params``.
            params: The parameters of the loss. If given as a list, the parameters are named "param_{i}"
                and the function does not take any arguments. If given as a dict, the function expects
                the parameter as the first argument (``params``).
        """
        if name is None:
            name = "SimpleConstraint"
        self._simple_func = func
        self._func_params = None
        if isinstance(params, collections.abc.Mapping):
            self._func_params = params
            params = list(params.values())
        self._simple_func_dependents = convert_to_container(
            params, container=OrderedSet
        )

        params = convert_to_container(params, container=list)
        if self._func_params is None:
            params = OrderedDict((f"param_{i}", p) for i, p in enumerate(params))
        else:
            params = self._func_params

        super().__init__(name=name, params=params)

    def _value(self):
        if self._func_params is None:
            return self._simple_func()
        else:
            return self._simple_func(self._func_params)


class ProbabilityConstraint(BaseConstraint):
    def __init__(
        self,
        observation: ztyping.NumericalScalarType | ZfitParameter,
        params: dict[str, ZfitParameter] = None,
        name: str = "ProbabilityConstraint",
        dtype=ztypes.float,
        **kwargs,
    ):
        """Base class for constraints using a probability density function.

        Args:
            dtype: the dtype of the constraint
            name: the name of the constraint
            params: The parameters to constraint
            observation: Observed values of the parameter
                to constraint obtained from auxiliary measurements.
        """
        # TODO: proper handling of input params, arrays. ArrayParam?
        params = convert_to_container(params)
        params_dict = {f"param_{i}": p for i, p in enumerate(params)}
        super().__init__(name=name, dtype=dtype, params=params_dict, **kwargs)
        params = tuple(self.params.values())

        observation = convert_to_container(observation, tuple)
        if len(observation) != len(params):
            raise ShapeIncompatibleError(
                "observation and params have to be the same length. Currently"
                f"observation: {len(observation)}, params: {len(params)}"
            )

        self._observation = observation  # TODO: needed below? Why?
        # for obs, p in zip(observation, params):
        #     obs = convert_to_parameter(obs, f"{p.name}_obs", prefer_constant=False)
        #     obs.floating = False
        #     self._observation.append(obs)

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
        """Sample ``n`` points from the probability density function for the observed value of the parameters.

        Args:
            n: The number of samples to be generated.
        Returns:
        """
        sample = self._sample(n=n)
        return {p: sample[:, i] for i, p in enumerate(self._ordered_params)}

    @abc.abstractmethod
    def _sample(self, n):
        raise NotImplementedError

    @property
    def _params_array(self):
        return znp.asarray(self._ordered_params)


class TFProbabilityConstraint(ProbabilityConstraint):
    def __init__(
        self,
        observation: ztyping.NumericalScalarType | ZfitParameter,
        params: dict[str, ZfitParameter],
        distribution: tfd.Distribution,
        dist_params,
        dist_kwargs=None,
        name: str = "DistributionConstraint",
        dtype=ztypes.float,
        **kwargs,
    ):
        """Base class for constraints using a probability density function from ``tensorflow_probability``.

        Args:
            distribution: The probability density function
                used to constraint the parameters
        """
        super().__init__(
            observation=observation, params=params, name=name, dtype=dtype, **kwargs
        )

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
        params = {k: tf.cast(v, ztypes.float) for k, v in params.items()}
        return self._distribution(**params, **kwargs, name=f"{self.name}_tfp")

    def _value(self):
        array = tf.cast(self._params_array, ztypes.float)
        value = -self.distribution.log_prob(array)
        return tf.reduce_sum(value)

    def _sample(self, n):
        return self.distribution.sample(n)


class GaussianConstraint(TFProbabilityConstraint, SerializableMixin):
    def __init__(
        self,
        params: ztyping.ParamTypeInput,
        observation: ztyping.NumericalScalarType,
        uncertainty: ztyping.NumericalScalarType,
    ):
        r"""Gaussian constraints on a list of parameters to some observed values with uncertainties.

        A Gaussian constraint is defined as the likelihood of ``params`` given the ``observations`` and ``uncertainty`` from
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
        uncertainty = convert_to_container(uncertainty, tuple)
        if (
            isinstance(uncertainty[0], (np.ndarray, tf.Tensor))
            and len(uncertainty) == 1
        ):
            uncertainty = tuple(uncertainty[0])
        original_init = {
            "observation": observation,
            "params": params,
            "uncertainty": uncertainty,
        }

        def create_covariance(mu, sigma):
            mu = z.convert_to_tensor(mu)
            sigma = np.asarray(
                sigma
            )  # otherwise TF complains that the shape got changed from [2] to [2, 2] (if we have a tuple of two arrays)
            sigma = z.convert_to_tensor(sigma)  # TODO (Mayou36): fix as above?
            params_tensor = z.convert_to_tensor(params)

            if sigma.shape.ndims > 1:
                covariance = sigma  # TODO: square as well?
            elif sigma.shape.ndims == 1:
                covariance = tf.linalg.tensor_diag(z.pow(sigma, 2.0))
            else:
                sigma = znp.reshape(sigma, [1])
                covariance = tf.linalg.tensor_diag(z.pow(sigma, 2.0))

            if (
                not params_tensor.shape[0]
                == mu.shape[0]
                == covariance.shape[0]
                == covariance.shape[1]
            ):
                raise ShapeIncompatibleError(
                    f"params_tensor, observation and uncertainty have to have the"
                    " same length. Currently"
                    f"param: {params_tensor.shape[0]}, mu: {mu.shape[0]}, "
                    f"covariance (from uncertainty): {covariance.shape[0:2]}"
                )
            return covariance

        distribution = tfd.MultivariateNormalTriL
        dist_params = lambda observation: dict(
            loc=observation,
            scale_tril=tf.linalg.cholesky(create_covariance(observation, uncertainty)),
        )
        dist_kwargs = dict(validate_args=True)

        super().__init__(
            name="GaussianConstraint",
            observation=observation,
            params=params,
            distribution=distribution,
            dist_params=dist_params,
            dist_kwargs=dist_kwargs,
        )
        self.hs3.original_init.update(original_init)
        self._covariance = lambda: create_covariance(self.observation, uncertainty)

    @property
    def covariance(self):
        """Return the covariance matrix of the observed values of the parameters constrained."""
        return self._covariance()


class GaussianConstraintRepr(BaseConstraintRepr):
    _implementation = GaussianConstraint
    hs3_type: Literal["GaussianConstraint"] = pydantic.Field(
        "GaussianConstraint", alias="type"
    )

    params: List[Serializer.types.ParamInputTypeDiscriminated]
    observation: List[Serializer.types.ParamInputTypeDiscriminated]
    uncertainty: List[Serializer.types.ParamInputTypeDiscriminated]

    @pydantic.root_validator(pre=True)
    def get_init_args(cls, values):
        if cls.orm_mode(values):
            values = values["hs3"].original_init
        return values

    @pydantic.validator("params", "observation", "uncertainty")
    def validate_params(cls, v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        else:
            v = convert_to_container(v, list)
        return v


class PoissonConstraint(TFProbabilityConstraint, SerializableMixin):
    def __init__(
        self, params: ztyping.ParamTypeInput, observation: ztyping.NumericalScalarType
    ):
        r"""Poisson constraints on a list of parameters to some observed values.

        Constraints parameters that can be counts (i.e. from a histogram) or, more generally, are
        Poisson distributed. This is often used in the case of histogram templates which are obtained
        from simulation and have a poisson uncertainty due to limited statistics.

        .. math::
            \text{constraint} = \text{Poisson}(\text{observation}; \text{params})


        Args:
            params: The parameters to constraint; corresponds to the mu in the Poisson
                distribution.
            observation: observed values of the parameter; corresponds to lambda
                in the Poisson distribution.
        Raises:
            ShapeIncompatibleError: If params and observation have incompatible shapes.
        """

        observation = convert_to_container(observation, tuple)
        params = convert_to_container(params, tuple)
        original_init = {"observation": observation, "params": params}

        distribution = tfd.Poisson
        dist_params = dict(rate=observation)
        dist_kwargs = dict(validate_args=False)

        super().__init__(
            name="PoissonConstraint",
            observation=observation,
            params=params,
            distribution=distribution,
            dist_params=dist_params,
            dist_kwargs=dist_kwargs,
        )
        self.hs3.original_init.update(original_init)


class PoissonConstraintRepr(BaseConstraintRepr):
    _implementation = PoissonConstraint
    hs3_type: Literal["PoissonConstraint"] = pydantic.Field(
        "PoissonConstraint", alias="type"
    )

    params: List[Serializer.types.ParamInputTypeDiscriminated]
    observation: List[Serializer.types.ParamInputTypeDiscriminated]

    @pydantic.root_validator(pre=True)
    def get_init_args(cls, values):
        if cls.orm_mode(values):
            values = values["hs3"].original_init
        return values

    @pydantic.validator("params", "observation")
    def validate_params(cls, v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        else:
            v = convert_to_container(v, list)
        return v


class LogNormalConstraint(TFProbabilityConstraint, SerializableMixin):
    def __init__(
        self,
        params: ztyping.ParamTypeInput,
        observation: ztyping.NumericalScalarType,
        uncertainty: ztyping.NumericalScalarType,
    ):
        r"""Log-normal constraints on a list of parameters to some observed values.

        Constraints parameters that can be counts (i.e. from a histogram) or, more generally, are
        LogNormal distributed. This is often used in the case of histogram templates which are obtained
        from simulation and have a log-normal uncertainty due to a multiplicative uncertainty.

        .. math::
            \text{constraint} = \text{LogNormal}(\text{observation}; \text{params})


        Args:
            params: The parameters to constraint; corresponds to the mu in the Poisson
                distribution.
            observation: observed values of the parameter; corresponds to lambda
                in the Poisson distribution.
            uncertainty: uncertainty of the observed values of the parameter; corresponds to sigma
                in the Poisson distribution.
        Raises:
            ShapeIncompatibleError: If params, mu and sigma have incompatible shapes.
        """

        observation = convert_to_container(observation, tuple)
        params = convert_to_container(params, tuple)
        uncertainty = convert_to_container(uncertainty, tuple)
        original_init = {
            "observation": observation,
            "params": params,
            "uncertainty": uncertainty,
        }

        distribution = tfd.LogNormal
        dist_params = lambda observation: dict(loc=observation, scale=uncertainty)
        dist_kwargs = dict(validate_args=False)

        super().__init__(
            name="LogNormalConstraint",
            observation=observation,
            params=params,
            distribution=distribution,
            dist_params=dist_params,
            dist_kwargs=dist_kwargs,
        )
        self.hs3.original_init.update(original_init)


class LogNormalConstraintRepr(BaseConstraintRepr):
    _implementation = LogNormalConstraint
    hs3_type: Literal["LogNormalConstraint"] = pydantic.Field(
        "LogNormalConstraint", alias="type"
    )

    params: List[Serializer.types.ParamInputTypeDiscriminated]
    observation: List[Serializer.types.ParamInputTypeDiscriminated]
    uncertainty: List[Serializer.types.ParamInputTypeDiscriminated]

    @pydantic.root_validator(pre=True)
    def get_init_args(cls, values):
        if cls.orm_mode(values):
            values = values["hs3"].original_init
        return values

    @pydantic.validator("params", "observation", "uncertainty")
    def validate_params(cls, v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        else:
            v = convert_to_container(v, list)
        return v
