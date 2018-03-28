from zfit.core import optimization as opt

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
ckm = [ lambda_CKM, A_CKM, rho_CKM, eta_CKM ]


# CKM parametrization taken from EOS
rho_eta = Complex(rho_CKM, eta_CKM) * CastComplex( Sqrt(1. - Square(A_CKM) * tf.pow(lambda_CKM,4)) / Sqrt(1.0 - Square(lambda_CKM)) ) / (1.
          - CastComplex( Square(A_CKM) * tf.pow(lambda_CKM,4) ) * Complex(rho_CKM, eta_CKM))

Vtb = 1. - Square(A_CKM) * tf.pow(lambda_CKM,4) / 2. - Square(A_CKM) * tf.pow(lambda_CKM,6) * AbsSq(rho_eta) / 2. - tf.pow(A_CKM,4) *  tf.pow(lambda_CKM,8) / 8.

Vts = CastComplex( -1.0 * A_CKM * Square(lambda_CKM) ) * (1.0 - CastComplex(Square(lambda_CKM)) * (1.0 - 2.0 * rho_eta) / 2.0 
      - CastComplex(tf.pow(lambda_CKM,4) / 8.0) - CastComplex(tf.pow(lambda_CKM,6))* (1.0 + 8.0 * CastComplex(Square(A_CKM)) * rho_eta) / 16.0 )
Vts = tf.abs(Vts)
