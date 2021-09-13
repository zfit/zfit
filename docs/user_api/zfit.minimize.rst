.. _minimize_user_api:

Minimize
--------

This module contains everything related to minimization in zfit.

Minimizers
##########


zfit supplies wrappers for different minimizers from multiple libraries. Most of the are local
minimizers (such as :class:`~zfit.minimize.Minuit`, :class:`~zfit.minimize.IpyoptV1` or
:class:`~zfit.minimize.ScipyLBFGSBV1` are) while there are also a few global ones such as
the :class:`~zfit.minimize.NLoptISRESV1` or :class:`~zfit.minimize.NLoptStoGOV1`.

.. toctree::
    :maxdepth: 2

    minimize/minimizers


Strategy
#############

Strategy to deal with NaNs and to provide callbacks.

.. toctree::
    :maxdepth: 2

    minimize/strategy



Criterion
#############

Criterion for the convergence of the minimization.

.. toctree::
    :maxdepth: 2

    minimize/criterion
