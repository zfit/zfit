Minimizers
###########

zfit supplies wrappers for different minimizers from multiple libraries. Most of the are local
minimizers (such as :class:`~zfit.minimize.Minuit`, :class:`~zfit.minimize.Ipyopt` or
:class:`~zfit.minimize.ScipyLBFGSB` are) while there are also a few global ones such as
the :class:`~zfit.minimize.NLoptISRES` or :class:`~zfit.minimize.NLoptStoGO`.

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
:::::::

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Minuit

Levenberg-Marquardt
:::::::::::::::::::::

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.LevenbergMarquardt


Ipyopt
:::::::


.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Ipyopt


Scipy
::::::

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.ScipyBFGS
    zfit.minimize.ScipyLBFGSB
    zfit.minimize.ScipyTrustConstr
    zfit.minimize.ScipyPowell
    zfit.minimize.ScipySLSQP
    zfit.minimize.ScipyTruncNC
    zfit.minimize.ScipyCOBYLA
    zfit.minimize.ScipyTrustNCG
    zfit.minimize.ScipyDogleg
    zfit.minimize.ScipyTrustKrylov




NLopt
::::::

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
