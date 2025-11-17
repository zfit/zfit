Prior distributions
-------------------

Prior distributions encode prior beliefs about parameters in Bayesian inference.
They represent knowledge or assumptions about parameter values before observing data.

.. currentmodule:: zfit.prior

Base classes
############

.. autosummary::
   :toctree: _generated/

   ZfitPrior
   AdaptivePrior

Parametric priors
##################

Common continuous probability distributions for parameters.

.. autosummary::
   :toctree: _generated/

   Normal
   Uniform
   HalfNormal
   Gamma
   Beta
   LogNormal
   Cauchy
   StudentT
   Exponential
   Poisson

Non-parametric priors
#####################

Empirical priors constructed from data samples.

.. autosummary::
   :toctree: _generated/

   KDE
