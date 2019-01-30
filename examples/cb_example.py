import numpy as np
import tensorflow as tf

import zfit
# Wrapper for some tensorflow functionality
from zfit import ztf

print("TensorFlow version:", tf.__version__)
Bmass = zfit.Space('M_ppiKS', limits=(5469, 5938))

# Lb -> KSppi
# path = "/Users/rsilvaco/Research/PosDoc/Analysis/Charmless/Lb2KSppiACP/data/MC/2011/1fb
# /Inv_KSppi_DD_Stripping17b_MCTruth_Trigger_NN.root"
path = "/home/jonas/Documents/physics/software/zfit_project/tmp/Inv_KSppi_DD_Stripping17b_MCTruth_Trigger_NN.root"
treepath = "DecayTree"
branches = ['M_ppiKS']
data = zfit.data.Data.from_root(path=path, treepath=treepath, branches=branches)

mu = zfit.Parameter("mu", 5619, 5500, 5650)
sigma = zfit.Parameter("sigma", 30, 0, 300)

a0 = zfit.Parameter("a0", 1., 0., 10.)
a1 = zfit.Parameter("a1", -1., -10., 0.)

n0 = zfit.Parameter("n0", 1, 0., 10.)
n1 = zfit.Parameter("n1", 1., 0., 10.)

frac = zfit.Parameter("frac", 0.5, 0., 1.)

# gauss = zfit.pdf.Gauss(obs=Bmass, mu=mu, sigma=sigma)
CB1 = zfit.pdf.CrystalBallPDF(obs=Bmass, mu=mu, sigma=sigma, alpha=a0, n=n0)
CB2 = zfit.pdf.CrystalBallPDF(obs=Bmass, mu=mu, sigma=sigma, alpha=a1, n=n1)

# Double Crystal Ball with a common mean and width
DoubleCB = zfit.pdf.SumPDF(pdfs=[CB1, CB2], fracs=frac)
# CB = frac*CB1 +CB2

# Create the negative log likelihood
from zfit.core.loss import UnbinnedNLL

nll = UnbinnedNLL(model=[DoubleCB], data=[data], fit_range=[Bmass])
minimize_params = [mu, sigma, a0, a1, n0]
# tf.gradients(nll.value(), minimize_params)

# Load and instantiate a tensorflow minimizer
from zfit.minimizers.minimizer_minuit import MinuitMinimizer

minimizer = MinuitMinimizer()

# Create the minimization graph to minimize
minimum = minimizer.minimize(loss=nll, params=minimize_params)

# Get the fitted values, again by run the variable graphs
# params = minimum.get_parameters()
params = minimum.params

print(params)
