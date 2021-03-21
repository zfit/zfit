Minimizers
----------

Zfit supplies wrappers for different minimizers from multiple libraries.


Root
====

.. autosummary::
    :toctree: minimizers

    zfit.minimize.Adam
    zfit.minimize.Minuit


Scipy
=====

.. autosummary::
    :toctree: minimizers

    zfit.minimize.ScipyLBFGSBV1
    zfit.minimize.ScipyTrustKrylovV1
    zfit.minimize.ScipyTrustConstrV1
    zfit.minimize.ScipyDoglegV1
    zfit.minimize.ScipyTrustNCGV1
    zfit.minimize.ScipyPowellV1
    zfit.minimize.ScipySLSQPV1
    zfit.minimize.ScipyNewtonCGV1
    zfit.minimize.ScipyTruncNCV1


NLopt
=====

.. autosummary::
    :toctree: minimizers

    zfit.minimize.NLoptLBFGSV1
    zfit.minimize.NLoptTruncNewtonV1
    zfit.minimize.NLoptSLSQPV1
    zfit.minimize.NLoptMMAV1
    zfit.minimize.NLoptCCSAQV1
    zfit.minimize.NLoptMLSLV1
    zfit.minimize.NLoptStoGOV1
    zfit.minimize.NLoptSubplexV1


Ipyopt
======


.. autosummary::
    :toctree: minimizers

    zfit.minimize.IpyoptV1


Tensorflow Probability
======================

.. autosummary::
    :toctree: minimizers

    zfit.minimize.BFGS
