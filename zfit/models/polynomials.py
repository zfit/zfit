"""Recurrent polynomials."""

from typing import Callable
import tensorflow as tf

import zfit


class RecursivePolynomial(zfit.func.BaseFunc):
    """1D polynomial generated via three-term recurrence.

    Args:
        degree (int): Degree of the polynomial to calculate
        f0 (Callable): Order 0 polynomial
        f1 (Callable): Order 1 polynomial
        Recurrence(func): Recurrence relation as function of the two previous
            polynomials and the degree n:

          x_{n+1} = recurrence(x_{n}, x_{n-1}, n)

    """

    def __init__(self, obs, degree: int, f0: Callable, f1: Callable, recurrence: Callable, name: str = "Polynomial", **kwargs):  # noqa
        self._degree = degree
        self._polys = [f0, f1]
        self._recurrence = recurrence
        super().__init__(obs=obs, name=name, **kwargs)

    @property
    def degree(self):
        """int: degree of the polynomial."""
        return self._degree

    def _func(self, x):
        x = x.unstack_x()
        polys = [self._polys[0](x), self._polys[1](x)]
        for i_deg in range(2, self.degree):
            polys.append(self.recurrence(polys[-1], polys[-2], i_deg, x))
        return polys[-1]


def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

    """
    return ((2 * n + 1) * tf.multiply(x, p1(x)) - n * p2(x)) / (n + 1)


class Legendre(RecursivePolynomial):
    """Legendre polynomials."""

    def __init__(self, obs, degree: int, name: str = "Legendre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=tf.ones_like, f1=lambda x: x,
                         recurrence=legendre_recurrence, **kwargs)


def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

    """
    return 2 * tf.multiply(x, p1(x)) - p2(x)


class Chebyshev(RecursivePolynomial):
    """Chebyshev polynomials."""

    def __init__(self, obs, degree: int, name: str = "Chebyshev", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=tf.ones_like, f1=lambda x: x,
                         recurrence=chebyshev_recurrence, **kwargs)


class Chebyshev2(RecursivePolynomial):
    """Chebyshev polynomials of the second kind."""

    def __init__(self, obs, degree: int, name: str = "Chebyshev2", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=tf.ones_like, f1=lambda x: x*2,
                         recurrence=chebyshev_recurrence, **kwargs)


def laguerre_recurrence(p1, p2, n, x):
    """Recurrence relation for Laguerre polynomials.

    (n+1) L_{n+1}(x) = (2n + 1 - x) L_{n}(x) - n L_{n-1}(x)

    """
    return (tf.multiply(2*n + 1 - x, p1(x)) - n * p2(x)) / (n+1)


class Laguerre(RecursivePolynomial):
    """Laguerre polynomials."""

    def __init__(self, obs, degree: int, name: str = "Laguerre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=tf.ones_like, f1=lambda x: 1 - x,
                         recurrence=laguerre_recurrence, **kwargs)


def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)

    """
    return 2 * (tf.multiply(x, p1(x)) - n * p2(x))


class Hermite(RecursivePolynomial):
    """Hermite polynomials as defined for Physics."""

    def __init__(self, obs, degree: int, name: str = "Hermite", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=tf.ones_like, f1=lambda x: 2*x,
                         recurrence=hermite_recurrence, **kwargs)


# EOF
