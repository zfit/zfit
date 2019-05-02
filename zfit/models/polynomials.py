#  Copyright (c) 2019 zfit
"""Recurrent polynomials."""

from typing import Callable
import tensorflow as tf

import zfit
from zfit import ztf
from ..core.limits import Space
from ..core.basepdf import BasePDF


class RecursivePolynomial(BasePDF):
    """1D polynomial generated via three-term recurrence.


    """

    def __init__(self, obs, degree: int, f0: Callable, f1: Callable, recurrence: Callable, name: str = "Polynomial",
                 **kwargs):  # noqa
        """

        Args:
            degree (int): Degree of the polynomial to calculate
            f0 (Callable): Order 0 polynomial
            f1 (Callable): Order 1 polynomial
            Recurrence(func): Recurrence relation as function of the two previous
                polynomials and the degree n:

                .. math::
                   x_{n+1} = recurrence(x_{n}, x_{n-1}, n)
        """
        self._degree = degree
        self._polys = [f0, f1]
        self._recurrence = recurrence
        super().__init__(obs=obs, name=name, **kwargs)

    @property
    def degree(self):
        """int: degree of the polynomial."""
        return self._degree

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        polys = self.do_recurrence(x, polys=self._polys, degree=self.degree, recurrence=self.recurrence)
        return polys[-1]

    @staticmethod
    def do_recurrence(x, polys, degree, recurrence):
        polys = [polys[0](x), polys[1](x)]
        for i_deg in range(2, degree):
            polys.append(recurrence(polys[-1], polys[-2], i_deg, x))
        return polys

    @property
    def recurrence(self):
        return self._recurrence


def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    .. math::
         (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

    """
    return ((2 * n + 1) * tf.multiply(x, p1) - n * p2) / (n + 1)


def legendre_integral(x, limits, norm_range, params, model):
    """Recursive integral of Legendre polynomials"""
    lower, upper = limits.limits
    lower = ztf.convert_to_tensor(lower)
    upper = ztf.convert_to_tensor(upper)
    if model.degree == 0:
        integral = limits.area()  # if polynomial 0 is 1

    else:
        def indefinite_integral(limits):
            if model.degree > 0:
                degree = model.degree + 1
            else:
                degree = model.degree
            polys = RecursivePolynomial.do_recurrence(x=limits, polys=model._polys, degree=degree,
                                                      recurrence=model.recurrence)

            one_limit_integral = (polys[-1] - polys[-3]) / (2. * ztf.convert_to_tensor(model.degree))
            return one_limit_integral

        integral = indefinite_integral(upper) - indefinite_integral(lower)
        integral = tf.reshape(integral, shape=())
    return integral


class Legendre(RecursivePolynomial):
    """Legendre polynomials."""

    def __init__(self, obs, degree: int, name: str = "Legendre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: x, degree=degree,
                         recurrence=legendre_recurrence, **kwargs)


legendre_limits = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Legendre.register_analytic_integral(func=legendre_integral, limits=legendre_limits)


def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

    """
    return 2 * tf.multiply(x, p1) - p2


class Chebyshev(RecursivePolynomial):
    """Chebyshev polynomials."""

    def __init__(self, obs, degree: int, name: str = "Chebyshev", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: x, degree=degree,
                         recurrence=chebyshev_recurrence, **kwargs)


def func_integral_chebyshev1(x, limits, norm_range, params, model):
    n = model.degree
    lower, upper = limits.limits
    lower = ztf.convert_to_tensor(lower)
    upper = ztf.convert_to_tensor(upper)

    def indefinite_integral_one(limits):
        polys = RecursivePolynomial.do_recurrence(x=limits, degree=n + 1, polys=model._polys,
                                                  recurrence=model.recurrence)
        n_float = ztf.convert_to_tensor(n)
        one_limits_integral = n_float * polys[-1] / (ztf.square(n_float) - 1) - limits[:, 0] * polys[-2] / (n_float - 1)
        return one_limits_integral

    integral = indefinite_integral_one(upper) - indefinite_integral_one(lower)
    integral = tf.reshape(integral, shape=())
    return integral


chebyshev1_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev.register_analytic_integral(func=func_integral_chebyshev1, limits=chebyshev1_limits_integral)


def func_integral_chebyshev1_one_to_one(x, limits, norm_range, params, model):
    if model.degree == 0:
        integral = 0
    else:
        integral = ((-1) ** model.degree + 1) / (1 - model.degree ** 2)

    integral = ztf.convert_to_tensor(integral)
    return integral


chebyshev1_limits_integral_one_to_one = Space.from_axes(axes=0, limits=(-1, 1))
Chebyshev.register_analytic_integral(func=func_integral_chebyshev1_one_to_one,
                                     limits=chebyshev1_limits_integral_one_to_one,
                                     priority=999)


class Chebyshev2(RecursivePolynomial):
    """Chebyshev polynomials of the second kind."""

    def __init__(self, obs, degree: int, name: str = "Chebyshev2", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: x * 2, degree=degree,
                         recurrence=chebyshev_recurrence, **kwargs)


def laguerre_recurrence(p1, p2, n, x):
    """Recurrence relation for Laguerre polynomials.

    (n+1) L_{n+1}(x) = (2n + 1 - x) L_{n}(x) - n L_{n-1}(x)

    """
    return (tf.multiply(2 * n + 1 - x, p1) - n * p2) / (n + 1)


class Laguerre(RecursivePolynomial):
    """Laguerre polynomials."""

    def __init__(self, obs, degree: int, name: str = "Laguerre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: 1 - x, degree=degree,
                         recurrence=laguerre_recurrence, **kwargs)


def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)

    """
    return 2 * (tf.multiply(x, p1) - n * p2)


class Hermite(RecursivePolynomial):
    """Hermite polynomials as defined for Physics."""

    def __init__(self, obs, degree: int, name: str = "Hermite", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: 2 * x, degree=degree,
                         recurrence=hermite_recurrence, **kwargs)

# EOF
