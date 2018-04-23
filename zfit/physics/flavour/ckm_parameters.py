from __future__ import absolute_import, division, print_function

import tensorflow as tf

from zfit.core.parameter import FitParameter
from zfit.core.tfext import abs_square, to_complex

# CKM parameters (from C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation))
lambda_CKM_init  =  0.225
A_CKM_init       =  0.829
rho_CKM_init     =  0.132
eta_CKM_init     =  0.348
lambda_CKM_sigma =  0.006
A_CKM_sigma      =  0.012
rho_CKM_sigma    =  0.018
eta_CKM_sigma    =  0.012


# CKM parameters
lambda_CKM  =  FitParameter("lambda_CKM", lambda_CKM_init, 0., 2., 0)
A_CKM       =  FitParameter("A_CKM", A_CKM_init, 0., 2., 0)
rho_CKM     =  FitParameter("rho_CKM", rho_CKM_init, 0., 2., 0)
eta_CKM     =  FitParameter("eta_CKM", eta_CKM_init, 0., 2., 0)

# CKM parameters
ckm = [lambda_CKM, A_CKM, rho_CKM, eta_CKM]

# CKM parametrization taken from EOS
rho_eta = (tf.complex(rho_CKM, eta_CKM) *
           to_complex(tf.sqrt(1. - tf.square(A_CKM) * tf.pow(lambda_CKM, 4)) /
                      tf.sqrt(1.0 - tf.square(lambda_CKM))) /
           (1. - to_complex(tf.square(A_CKM) * tf.pow(lambda_CKM, 4)) * tf.complex(rho_CKM,
                                                                                   eta_CKM)))

Vtb = (1. - tf.square(A_CKM) * tf.pow(lambda_CKM, 4) / 2. - tf.square(A_CKM) * tf.pow(lambda_CKM,
                                                                                      6) *
       abs_square(rho_eta) / 2. - tf.pow(A_CKM, 4) * tf.pow(lambda_CKM, 8) / 8.)

Vts = (to_complex(-1.0 * A_CKM * tf.square(lambda_CKM)) *
       (1.0 - to_complex(tf.square(lambda_CKM)) * (1.0 - 2.0 * rho_eta) / 2.0 -
        to_complex(tf.pow(lambda_CKM, 4) / 8.0) - to_complex(tf.pow(lambda_CKM, 6)) *
        (1.0 + 8.0 * to_complex(tf.square(A_CKM)) * rho_eta) / 16.0))
Vts = tf.abs(Vts)
