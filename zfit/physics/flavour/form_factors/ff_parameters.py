from __future__ import absolute_import, division, print_function

from zfit.core.parameter import FitParameter
from .data import priors_ff

# Form Factors for B0 -> K* ll

A0_0  =  FitParameter("A0_0", priors_ff.param_mean_FF[0], -10., 10., 0)
A0_1  =  FitParameter("A0_1", priors_ff.param_mean_FF[1], -10., 10., 0)
A0_2  =  FitParameter("A0_2", priors_ff.param_mean_FF[2], -15., 15., 0)
A1_0  =  FitParameter("A1_0", priors_ff.param_mean_FF[3], -10., 10., 0)
A1_1  =  FitParameter("A1_1", priors_ff.param_mean_FF[4], -10., 10., 0)
A1_2  =  FitParameter("A1_2", priors_ff.param_mean_FF[5], -10., 10., 0)
A12_1 =  FitParameter("A12_1", priors_ff.param_mean_FF[6], -10., 10., 0)
A12_2 =  FitParameter("A12_2", priors_ff.param_mean_FF[7], -10., 10., 0)
V_0   =  FitParameter("V_0", priors_ff.param_mean_FF[8], -10., 10., 0)
V_1   =  FitParameter("V_1", priors_ff.param_mean_FF[9], -10., 10., 0)
V_2   =  FitParameter("V_2", priors_ff.param_mean_FF[10], -15., 15., 0)
T1_0  =  FitParameter("T1_0", priors_ff.param_mean_FF[11], -10., 10., 0)
T1_1  =  FitParameter("T1_1", priors_ff.param_mean_FF[12], -10., 10., 0)
T1_2  =  FitParameter("T1_2", priors_ff.param_mean_FF[13], -15., 15., 0)
T2_1  =  FitParameter("T2_1", priors_ff.param_mean_FF[14], -10., 10., 0)
T2_2  =  FitParameter("T2_2", priors_ff.param_mean_FF[15], -10., 10., 0)
T23_0 =  FitParameter("T23_0", priors_ff.param_mean_FF[16], -10., 10., 0)
T23_1 =  FitParameter("T23_1", priors_ff.param_mean_FF[17], -10., 10., 0)
T23_2 =  FitParameter("T23_2", priors_ff.param_mean_FF[18], -20., 20., 0)

# Form factors
ff = [ A0_0, A0_1, A0_2, A1_0, A1_1, A1_2, A12_1, A12_2, V_0, V_1, V_2, T1_0, T1_1, T1_2, T2_1, T2_2, T23_0, T23_1, T23_2 ]
