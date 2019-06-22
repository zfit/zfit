import abc

from typing import Dict, Union, Callable, Optional

from .core.baseobject import BaseNumeric
from .core.interfaces import ZfitParameter
from .util import ztyping
from .util.graph import get_dependents_auto
from .util.container import convert_to_container
from .util.exception import ShapeIncompatibleError
from ..settings import ztypes
from zfit import ztf

import tensorflow as tf
import numpy as np
import tensorflow_probability.distributions as tfd
mvnfc = tfd.MultivariateNormalFullCovariance


class BaseConstraint(BaseNumeric):

    def __init__(self, params: Union[Dict[str, ZfitParameter], None] = None,
                 name: str = "BaseModel", dtype=ztypes.float,
                 **kwargs):

        super().__init__(name=name, dtype=dtype, params=params, **kwargs)

    @abc.abstractmethod
    def _eval(self):
        raise NotImplementedError

    def value(self):
        return self._value()

    def _value(self):
        try:
            return self._eval()

        except NotImplementedError:
            raise NotImplementedError("_eval not properly defined!")

    def sample(self):
        return self._sample

    @abc.abstractmethod
    def _sample(self):
        raise NotImplementedError()

    def _get_dependents(self) -> ztyping.DependentsType:
        return self._extract_dependents(self.get_params())


class SimpleConstraint(BaseConstraint):

    def __init__(self, func: Callable, params: Optional[ztyping.ParametersType] = None):

        self._simple_func = func
        self._simple_func_dependents = convert_to_container(params, container=set)
        if params is None:
            params = convert_to_container(self.get_dependents(), container=list)
        params = {p.name: p for p in params}

        super().__init__(name="SimpleConstraint", params=params)

    def _get_dependents(self):
        dependents = self._simple_func_dependents
        if dependents is None:
            independent_params = tf.get_collection("zfit_independent")
            dependents = get_dependents_auto(tensor=self.value(), candidates=independent_params)
            self._simple_func_dependents = dependents
        return dependents

    def _eval(self):
        return self._simple_func()


class GaussianConstraint(BaseConstraint):

    def __init__(self, params: ztyping.ParamTypeInput, mu: ztyping.NumericalScalarType,
                 sigma: ztyping.NumericalScalarType):

        params = convert_to_container(params, tuple)
        mu = convert_to_container(mu, container=tuple, non_containers=[np.ndarray])
        super().__init__(name="GaussianConstraint", params={p.name: p for p in params})

        params = ztf.convert_to_tensor(params)
        mu = ztf.convert_to_tensor(mu)
        sigma = ztf.convert_to_tensor(sigma)

        if sigma.shape.ndims > 1:
            covariance = sigma
        elif sigma.shape.ndims == 1:
            covariance = tf.diag(ztf.pow(sigma, 2.))
        else:
            sigma = tf.reshape(sigma, [1])
            covariance = tf.diag(ztf.pow(sigma, 2.))

        if not params.shape[0] == mu.shape[0] == covariance.shape[0] == covariance.shape[1]:
            raise ShapeIncompatibleError(f"params, mu and sigma have to have the same length. Currently"
                                         f"param: {params.shape[0]}, mu: {mu.shape[0]}, "
                                         f"covariance (from sigma): {covariance.shape[0:2]}")

        self._covariance = covariance
        self._mu = mu
        self._tparams = params
        self._dist = mvnfc(loc=mu, covariance_matrix=covariance, validate_args=True)

    def _eval(self):
        value = -self._dist.log_prob(self._tparams)
        return value

    def _sample(self):
        sample = self._dist.sample()
        return {p: sample[i] for i, p in enumerate(self.get_params())}
