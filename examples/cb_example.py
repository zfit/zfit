import numpy as np
import tensorflow as tf

import zfit
# Wrapper for some tensorflow functionality
from zfit import ztf
from zfit.minimizers.minimizers_scipy import ScipyMinimizer

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
# mu = zfit.Parameter("mu", 5619)
mu2 = zfit.Parameter("mu2", 5619, 5500, 5650)
# mu2 = zfit.Parameter("mu2", 5619)
sigma = zfit.Parameter("sigma", 60, 0, 300)
# sigma = zfit.Parameter("sigma", 60)
sigma2 = zfit.Parameter("sigma2", 60, 0, 300)
# sigma2 = zfit.Parameter("sigma2", 60)

a0 = zfit.Parameter("a0", 1., 0., 1000.)
# a0 = zfit.Parameter("a0", 1.)
a1 = zfit.Parameter("a1", -1., -10., 0.)
# a1 = zfit.Parameter("a1", -1.)

n0 = zfit.Parameter("n0", 1, 0, 10.)
# n0 = zfit.Parameter("n0", 1)
n1 = zfit.Parameter("n1", 1., 0000001., 10.)
# n1 = zfit.Parameter("n1", 1)

frac = zfit.Parameter("frac", 0.5, 0., 1.)

# gauss = zfit.pdf.Gauss(obs=Bmass, mu=mu, sigma=sigma)
CB1 = zfit.pdf.CrystalBallPDF(obs=Bmass, mu=mu, sigma=sigma, alpha=a0, n=n0)
CB2 = zfit.pdf.CrystalBallPDF(obs=Bmass, mu=mu, sigma=sigma, alpha=a1, n=n1)
# CB1 = zfit.pdf.Gauss(obs=Bmass, mu=mu, sigma=sigma)
# CB2 = zfit.pdf.Gauss(obs=Bmass, mu=mu2, sigma=sigma2)

# Double Crystal Ball with a common mean and width
DoubleCB = zfit.pdf.SumPDF(pdfs=[CB1, CB2], fracs=frac)
# DoubleCB = CB1
# CB = frac*CB1 +CB2

# Create the negative log likelihood
from zfit.core.loss import UnbinnedNLL

probs = DoubleCB.pdf(x=np.linspace(5469, 5938, num=1000))

nll = UnbinnedNLL(model=[DoubleCB], data=[data], fit_range=[Bmass])
# minimize_params = [mu, sigma, a0, a1]
minimize_params = None
# tf.gradients(nll.value(), minimize_params)
# nll.value()
# tf.add_check_numerics_ops()

# Load and instantiate a tensorflow minimizer
from zfit.minimizers.minimizer_minuit import MinuitMinimizer

# minimizer = MinuitMinimizer()
minimizer = ScipyMinimizer()

# Create the minimization graph to minimize
minimum = minimizer.minimize(loss=nll, params=minimize_params)

# Get the fitted values, again by run the variable graphs
# params = minimum.get_params()
params = minimum.params

# error = minimum.error()
# print(error)

print(params)
print(minimum.info)
