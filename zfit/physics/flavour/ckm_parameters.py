from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core import optimization as opt
from zfit.core.tfext import CastComplex, AbsSq

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
lambda_CKM  =  opt.FitParameter("lambda_CKM" , lambda_CKM_init , 0., 2., 0)
A_CKM       =  opt.FitParameter("A_CKM"      , A_CKM_init      , 0., 2., 0)
rho_CKM     =  opt.FitParameter("rho_CKM"    , rho_CKM_init    , 0., 2., 0)
eta_CKM     =  opt.FitParameter("eta_CKM"    , eta_CKM_init    , 0., 2., 0)

# CKM parameters
ckm = [lambda_CKM, A_CKM, rho_CKM, eta_CKM]

# CKM parametrization taken from EOS
rho_eta = (tf.complex(rho_CKM, eta_CKM) *
           CastComplex(tf.sqrt(1. - tf.square(A_CKM) * tf.pow(lambda_CKM, 4)) /
                       tf.sqrt(1.0 - tf.square(lambda_CKM))) /
           (1. - CastComplex(tf.square(A_CKM) * tf.pow(lambda_CKM, 4)) * tf.complex(rho_CKM,
                                                                                    eta_CKM)))

Vtb = (1. - tf.square(A_CKM) * tf.pow(lambda_CKM, 4) / 2. - tf.square(A_CKM) * tf.pow(lambda_CKM,
                                                                                      6) *
       AbsSq(rho_eta) / 2. - tf.pow(A_CKM, 4) * tf.pow(lambda_CKM, 8) / 8.)

Vts = (CastComplex(-1.0 * A_CKM * tf.square(lambda_CKM)) *
       (1.0 - CastComplex(tf.square(lambda_CKM)) * (1.0 - 2.0 * rho_eta) / 2.0 -
        CastComplex(tf.pow(lambda_CKM, 4) / 8.0) - CastComplex(tf.pow(lambda_CKM, 6)) *
        (1.0 + 8.0 * CastComplex(tf.square(A_CKM)) * rho_eta) / 16.0))
Vts = tf.abs(Vts)
