
zfit supplies wrappers for different minimizers from multiple libraries. Most of the are local
minimizers (such as :class:`~zfit.minimize.Minuit`, :class:`~zfit.minimize.IpyoptV1` or
:class:`~zfit.minimize.ScipyLBFGSBV1` are) while there are also a few global ones such as
the :class:`~zfit.minimize.NLoptISRESV1` or :class:`~zfit.minimize.NLoptStoGOV1`.

While the former are usually faster and preferred, they depend more on the initial values than
the latter. Especially in higher dimensions, a global search of the parameters
can increase the minimization time drastically and is often infeasible. It is also possible to
couple the minimizers by first doing an approximate global minimization and then polish the
minimum found with a local minimizer.

All minimizers support similar arguments, most notably ``tol`` which denotes the termination
value. This is reached if the value of the convergence criterion, which defaults to
:class:`~zfit.minimize.EDM`, the same that is also used in :class:`~zfit.minimize.Minuit`.

Other than that, there are a also a few minimizer specific arguments that differ from each minimizer.

They all have the exact same minimization method :meth:`~zfit.minimize.BaseMinimizer.minimize`
which takes a loss, parameters and (optionally) a :class:`~zfit.result.FitResult` from which it can
take information to have a better start into the minimization.

Minuit
======

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Minuit

Ipyopt
======


.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.IpyoptV1


Scipy
=====

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.ScipyLBFGSBV1
    zfit.minimize.ScipyTrustConstrV1
    zfit.minimize.ScipyPowellV1
    zfit.minimize.ScipySLSQPV1
    zfit.minimize.ScipyTruncNCV1


NLopt
=====

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.NLoptLBFGSV1
    zfit.minimize.NLoptTruncNewtonV1
    zfit.minimize.NLoptSLSQPV1
    zfit.minimize.NLoptMMAV1
    zfit.minimize.NLoptCCSAQV1
    zfit.minimize.NLoptSubplexV1
    zfit.minimize.NLoptCOBYLAV1
    zfit.minimize.NLoptMLSLV1
    zfit.minimize.NLoptStoGOV1
    zfit.minimize.NLoptBOBYQAV1
    zfit.minimize.NLoptISRESV1
    zfit.minimize.NLoptESCHV1
    zfit.minimize.NLoptShiftVarV1



Tensorflow
======================

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Adam
