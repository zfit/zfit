"""Numpy like interface for math functions and arrays. This module is intended to replace tensorflow specific methods
and datastructures with equivalent or similar versions in the numpy api. This should help make zfit as a project
portable to alternatives of tensorflow should it be necessary in the future. At the moment it is simply an alias for the
numpy api of tensorflow. See https://www.tensorflow.org/guide/tf_numpy for more a guide to numpy api in tensorflow. See
https://www.tensorflow.org/api_docs/python/tf/experimental/numpy for the complete numpy api in tensorflow. Recommended
way of importing:

>>> import zfit.z.numpy as znp
"""

#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.experimental.numpy import *  # noqa: F403

# ruff: noqa: F405


class linalg:
    inv = staticmethod(tf.linalg.inv)
    det = staticmethod(tf.linalg.det)
    solve = staticmethod(tf.linalg.solve)


# TODO: move into special namespace when that's available
def faddeeva_humlicek(z, s=10.0):
    """Complex error function w(z = x + iy) combining Humlicek's rational approximations.

    |x| + y > s:  Humlicek (JQSRT, 1982) rational approximation for region II;
    else:          Humlicek (JQSRT, 1979) rational approximation with n=16 and delta=y0=1.35

    Version using tensorflow and the numpy api of tensorflow.
    Single complex argument version of Franz Schreier's cpfX.hum2zpf16m.
    Originally licensed under a 3-clause BSD style license - see
    https://atmos.eoc.dlr.de/tools/lbl4IR/cpfX.py

    Args:
        z: array_like
        s: float, optional
            The threshold for the region II approximation.
            The default is 10.0.

    Returns:
        A `Tensor` of `complex128` dtype.
    """

    z = asarray(atleast_1d(z), dtype=complex128)

    AA = array(
        [
            +46236.3358828121,
            -147726.58393079657j,
            -206562.80451354137,
            281369.1590631087j,
            +183092.74968253175,
            -184787.96830696272j,
            -66155.39578477248,
            57778.05827983565j,
            +11682.770904216826,
            -9442.402767960672j,
            -1052.8438624933142,
            814.0996198624186j,
            +45.94499030751872,
            -34.59751573708725j,
            -0.7616559377907136,
            0.5641895835476449j,
        ],
        dtype=complex128,
    )

    bb = array(
        [
            +7918.06640624997,
            -126689.0625,
            +295607.8125,
            -236486.25,
            +84459.375,
            -15015.0,
            +1365.0,
            -60.0,
            +1.0,
        ],
        dtype=complex128,
    )

    sqrt_piinv = 1.0 / np.sqrt(np.pi)

    zz = z * z
    w = 1j * z * (zz * sqrt_piinv - 1.410474) / (0.75 + zz * (zz - 3.0))

    real_part = real(z)
    imag_part = imag(z)
    mask = logical_and(abs(real_part) + imag_part < s, imag_part < s)

    Z = z + 1.35j
    ZZ = Z * Z

    numer = (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((((AA[15] * Z + AA[14]) * Z + AA[13]) * Z + AA[12]) * Z + AA[11]) * Z
                                            + AA[10]
                                        )
                                        * Z
                                        + AA[9]
                                    )
                                    * Z
                                    + AA[8]
                                )
                                * Z
                                + AA[7]
                            )
                            * Z
                            + AA[6]
                        )
                        * Z
                        + AA[5]
                    )
                    * Z
                    + AA[4]
                )
                * Z
                + AA[3]
            )
            * Z
            + AA[2]
        )
        * Z
        + AA[1]
    ) * Z + AA[0]

    denom = (
        ((((((ZZ + bb[7]) * ZZ + bb[6]) * ZZ + bb[5]) * ZZ + bb[4]) * ZZ + bb[3]) * ZZ + bb[2]) * ZZ + bb[1]
    ) * ZZ + bb[0]

    return where(mask, numer / denom, w)
