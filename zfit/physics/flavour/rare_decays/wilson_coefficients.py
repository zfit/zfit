from __future__ import print_function, division, absolute_import

from zfit.core import optimization as opt

# Fit parameters: Wilson coefficients
ReC7     = opt.FitParameter("ReC7"    , -0.33726473   ,  -5., 5., 0)
ImC7     = opt.FitParameter("ImC7"    ,  0.           ,  -5., 5., 0)
ReC7p    = opt.FitParameter("ReC7p"   ,  0.           ,  -5., 5., 0)
ImC7p    = opt.FitParameter("ImC7p"   ,  0.           ,  -5., 5., 0)

ReC9   = opt.FitParameter("ReC9"  ,  4.27342842   ,  -20., 20., 0)
ImC9   = opt.FitParameter("ImC9"  ,  0.           ,   -5.,  5., 0)
ReC10  = opt.FitParameter("ReC10" , -4.16611761   ,  -20., 20., 0)
ImC10  = opt.FitParameter("ImC10" ,  0.           ,   -5.,  5., 0)
ReC9p  = opt.FitParameter("ReC9p" ,  0.           ,  -10., 10., 0)
ImC9p  = opt.FitParameter("ImC9p" ,  0.           ,   -5.,  5., 0)
ReC10p = opt.FitParameter("ReC10p",  0.           ,  -10., 10., 0)
ImC10p = opt.FitParameter("ImC10p",  0.           ,   -5.,  5., 0)


# define Wilson coeff.
C7   = tf.complex(ReC7  , ImC7  )
C7p  = tf.complex(ReC7p , ImC7p )
C9   = tf.complex(ReC9  , ImC9  )
C9p  = tf.complex(ReC9p , ImC9p )
C10  = tf.complex(ReC10 , ImC10 )
C10p = tf.complex(ReC10p, ImC10p)
