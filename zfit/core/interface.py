from __future__ import absolute_import, division, print_function

import numpy as np

from . import tfext


# Pi
def Pi(): return tfext.constant(np.pi)


# Return argument of a complex number


def Clebsch(j1, m1, j2, m2, J, M):
    """
      Return Clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
      (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """
    from sympy.physics.quantum.cg import CG
    from sympy import Rational
    return CG(Rational(j1, 2), Rational(m1, 2), Rational(j2, 2), Rational(m2, 2), Rational(J, 2),
              Rational(M, 2)).doit().evalf()


def SetSeed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)
