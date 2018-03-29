from __future__ import print_function, division, absolute_import

from zfit.core import optimization as opt
from .data import priors_nlh

# Parametrization of non-local hadronix effects in B0 -> K* ll

Re_alpha_perp_0 = opt.FitParameter("Re_alpha_perp_0", priors_nlh.param_mean_H[0] , -0.01, 0.01, 0)
Re_alpha_para_0 = opt.FitParameter("Re_alpha_para_0", priors_nlh.param_mean_H[1] , -0.01, 0.01, 0)
Re_alpha_zero_0 = opt.FitParameter("Re_alpha_zero_0", priors_nlh.param_mean_H[2] , -0.01, 0.01, 0)
Im_alpha_perp_0 = opt.FitParameter("Im_alpha_perp_0", priors_nlh.param_mean_H[3] , -0.01, 0.01, 0)
Im_alpha_para_0 = opt.FitParameter("Im_alpha_para_0", priors_nlh.param_mean_H[4] , -0.01, 0.01, 0)
Im_alpha_zero_0 = opt.FitParameter("Im_alpha_zero_0", priors_nlh.param_mean_H[5] , -0.01, 0.01, 0)

Re_alpha_perp_1 = opt.FitParameter("Re_alpha_perp_1", priors_nlh.param_mean_H[6] , -0.01, 0.01, 0)
Re_alpha_para_1 = opt.FitParameter("Re_alpha_para_1", priors_nlh.param_mean_H[7] , -0.01, 0.01, 0)
Re_alpha_zero_1 = opt.FitParameter("Re_alpha_zero_1", priors_nlh.param_mean_H[8] , -0.01, 0.01, 0)
Im_alpha_perp_1 = opt.FitParameter("Im_alpha_perp_1", priors_nlh.param_mean_H[9] , -0.01, 0.01, 0)
Im_alpha_para_1 = opt.FitParameter("Im_alpha_para_1", priors_nlh.param_mean_H[10], -0.01, 0.01, 0)
Im_alpha_zero_1 = opt.FitParameter("Im_alpha_zero_1", priors_nlh.param_mean_H[11], -0.01, 0.01, 0)

Re_alpha_perp_2 = opt.FitParameter("Re_alpha_perp_2", priors_nlh.param_mean_H[12], -0.01, 0.01, 0)
Re_alpha_para_2 = opt.FitParameter("Re_alpha_para_2", priors_nlh.param_mean_H[13], -0.01, 0.01, 0)
Im_alpha_perp_2 = opt.FitParameter("Im_alpha_perp_2", priors_nlh.param_mean_H[14], -0.01, 0.01, 0)
Im_alpha_para_2 = opt.FitParameter("Im_alpha_para_2", priors_nlh.param_mean_H[15], -0.01, 0.01, 0)

Re_alpha_zero_2 = opt.FitParameter("Re_alpha_zero_2", 0. , -0.1, 0.1, 0)
Im_alpha_zero_2 = opt.FitParameter("Im_alpha_zero_2", 0. , -0.1, 0.1, 0)

Re_alpha_perp_3 = opt.FitParameter("Re_alpha_perp_3", 0. , -0.1, 0.1, 0)
Re_alpha_para_3 = opt.FitParameter("Re_alpha_para_3", 0. , -0.1, 0.1, 0)
Im_alpha_perp_3 = opt.FitParameter("Im_alpha_perp_3", 0. , -0.1, 0.1, 0)
Im_alpha_para_3 = opt.FitParameter("Im_alpha_para_3", 0. , -0.1, 0.1, 0)
Re_alpha_zero_3 = opt.FitParameter("Re_alpha_zero_3", 0. , -1. , 1. , 0)
Im_alpha_zero_3 = opt.FitParameter("Im_alpha_zero_3", 0. , -1. , 1. , 0)

Re_alpha_perp_4 = opt.FitParameter("Re_alpha_perp_4", 0. , -1. , 1. , 0)
Re_alpha_para_4 = opt.FitParameter("Re_alpha_para_4", 0. , -1. , 1. , 0)
Im_alpha_perp_4 = opt.FitParameter("Im_alpha_perp_4", 0. , -1. , 1. , 0)
Im_alpha_para_4 = opt.FitParameter("Im_alpha_para_4", 0. , -1. , 1. , 0)
Re_alpha_zero_4 = opt.FitParameter("Re_alpha_zero_4", 0. , -1. , 1. , 0)
Im_alpha_zero_4 = opt.FitParameter("Im_alpha_zero_4", 0. , -1. , 1. , 0)

Re_alpha_perp_5 = opt.FitParameter("Re_alpha_perp_5", 0. , -1. , 1. , 0)
Re_alpha_para_5 = opt.FitParameter("Re_alpha_para_5", 0. , -1. , 1. , 0)
Im_alpha_perp_5 = opt.FitParameter("Im_alpha_perp_5", 0. , -1. , 1. , 0)
Im_alpha_para_5 = opt.FitParameter("Im_alpha_para_5", 0. , -1. , 1. , 0)


# Define complex parameters alphas
alpha_perp_0 = tf.complex(Re_alpha_perp_0, Im_alpha_perp_0)
alpha_para_0 = tf.complex(Re_alpha_para_0, Im_alpha_para_0)
alpha_zero_0 = tf.complex(Re_alpha_zero_0, Im_alpha_zero_0)

alpha_perp_1 = tf.complex(Re_alpha_perp_1, Im_alpha_perp_1)
alpha_para_1 = tf.complex(Re_alpha_para_1, Im_alpha_para_1)
alpha_zero_1 = tf.complex(Re_alpha_zero_1, Im_alpha_zero_1)

alpha_perp_2 = tf.complex(Re_alpha_perp_2, Im_alpha_perp_2)
alpha_para_2 = tf.complex(Re_alpha_para_2, Im_alpha_para_2)
alpha_zero_2 = tf.complex(Re_alpha_zero_2, Im_alpha_zero_2)

alpha_perp_3 = tf.complex(Re_alpha_perp_3, Im_alpha_perp_3)
alpha_para_3 = tf.complex(Re_alpha_para_3, Im_alpha_para_3)
alpha_zero_3 = tf.complex(Re_alpha_zero_3, Im_alpha_zero_3)

alpha_perp_4 = tf.complex(Re_alpha_perp_4, Im_alpha_perp_4)
alpha_para_4 = tf.complex(Re_alpha_para_4, Im_alpha_para_4)
alpha_zero_4 = tf.complex(Re_alpha_zero_4, Im_alpha_zero_4)

alpha_perp_5 = tf.complex(Re_alpha_perp_5, Im_alpha_perp_5)
alpha_para_5 = tf.complex(Re_alpha_para_5, Im_alpha_para_5)




### Define list of parameters

# All hadronic coefficients
alphas = [ Re_alpha_perp_0,  Re_alpha_para_0,  Re_alpha_zero_0,
           Im_alpha_perp_0,  Im_alpha_para_0,  Im_alpha_zero_0,
           Re_alpha_perp_1,  Re_alpha_para_1,  Re_alpha_zero_1,
           Im_alpha_perp_1,  Im_alpha_para_1,  Im_alpha_zero_1,
           Re_alpha_perp_2,  Re_alpha_para_2,  Re_alpha_zero_2,
           Im_alpha_perp_2,  Im_alpha_para_2,  Im_alpha_zero_2,
           Re_alpha_perp_3,  Re_alpha_para_3,  Re_alpha_zero_3,
           Im_alpha_perp_3,  Im_alpha_para_3,  Im_alpha_zero_3,
           Re_alpha_perp_4,  Re_alpha_para_4,  Re_alpha_zero_4,
           Im_alpha_perp_4,  Im_alpha_para_4,  Im_alpha_zero_4,
           Re_alpha_perp_5,  Re_alpha_para_5,
           Im_alpha_perp_5,  Im_alpha_para_5,
         ]

# Hadronic coefficients that corresponds to the default configuration (z^2 for perp and para, z^1 for zero)
alphaUpTo2 = [ Re_alpha_perp_0,  Re_alpha_para_0,  Re_alpha_zero_0,
               Im_alpha_perp_0,  Im_alpha_para_0,  Im_alpha_zero_0,
               Re_alpha_perp_1,  Re_alpha_para_1,  Re_alpha_zero_1,
               Im_alpha_perp_1,  Im_alpha_para_1,  Im_alpha_zero_1,
               Re_alpha_perp_2,  Re_alpha_para_2,
               Im_alpha_perp_2,  Im_alpha_para_2
             ]

# Hadronic coefficients that corresponds to the last order of the z^2 (default) fit -- [needed if I want to set the alpha(3) = +-k*alpha(2)]
alpha2 = [ Re_alpha_zero_1,  Im_alpha_zero_1,  Re_alpha_perp_2,  Re_alpha_para_2,  Im_alpha_perp_2,  Im_alpha_para_2 ]

# Hadronic coefficients that are added to the fit if mode alpha3 is required (z^3 for perp and para, z^2 for zero)
alpha3 = [ Re_alpha_zero_2,  Im_alpha_zero_2,  Re_alpha_perp_3,  Re_alpha_para_3,  Im_alpha_perp_3,  Im_alpha_para_3 ]

# Hadronic coefficients that are added to the fit if mode alpha4 is required (z^4 for all polarizations)
alpha4 = [ Re_alpha_zero_3,  Im_alpha_zero_3,  Re_alpha_perp_4,  Re_alpha_para_4,  Im_alpha_perp_4,  Im_alpha_para_4 ]

# Hadronic coefficients that are added to the fit if mode alpha5 is required (z^5 for all polarizations)
alpha5 = [ Re_alpha_zero_4,  Im_alpha_zero_4,  Re_alpha_perp_5,  Re_alpha_para_5,  Im_alpha_perp_5,  Im_alpha_para_5 ]
