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

Minimizer Comparison
-------------------

The following visualization shows how different minimizers perform on a complex version of the Rosenbrock function.
Each minimizer starts from 5 different initial points (marked with symbols) and follows a path to the minimum.
The contour plot represents the function values, with darker colors indicating lower values.
For each minimizer, the plot shows the number of function evaluations, gradient calculations, and execution time.


Minuit
:::::::

.. image:: ../../images/_generated/minimizers/minuit_paths.png
   :width: 100%
   :alt: Minuit minimizer paths on a complex Rosenbrock function

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Minuit

Levenberg-Marquardt
:::::::::::::::::::::

.. image:: ../../images/_generated/minimizers/levenbergmarquardt_paths.png
   :width: 100%
   :alt: LevenbergMarquardt minimizer paths on a complex Rosenbrock function

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.LevenbergMarquardt


Ipyopt
:::::::

.. image:: ../../images/_generated/minimizers/ipyopt_paths.png
   :width: 100%
   :alt: Ipyopt minimizer paths on a complex Rosenbrock function

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Ipyopt


Scipy
::::::

The following visualizations show how different Scipy minimizers perform on the complex Rosenbrock function:

.. image:: ../../images/_generated/minimizers/scipybfgs_paths.png
   :width: 100%
   :alt: ScipyBFGS minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipylbfgsb_paths.png
   :width: 100%
   :alt: ScipyLBFGSB minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustconstr_paths.png
   :width: 100%
   :alt: ScipyTrustConstr minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipypowell_paths.png
   :width: 100%
   :alt: ScipyPowell minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipyslsqp_paths.png
   :width: 100%
   :alt: ScipySLSQP minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytruncnc_paths.png
   :width: 100%
   :alt: ScipyTruncNC minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipycobyla_paths.png
   :width: 100%
   :alt: ScipyCOBYLA minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustncg_paths.png
   :width: 100%
   :alt: ScipyTrustNCG minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipydogleg_paths.png
   :width: 100%
   :alt: ScipyDogleg minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustkrylov_paths.png
   :width: 100%
   :alt: ScipyTrustKrylov minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipynewtoncg_paths.png
   :width: 100%
   :alt: ScipyNewtonCG minimizer paths on a complex Rosenbrock function

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
    zfit.minimize.ScipyNewtonCG




NLopt
::::::

The following visualizations show how different NLopt minimizers perform on the complex Rosenbrock function:

.. image:: ../../images/_generated/minimizers/nloptlbfgsv1_paths.png
   :width: 100%
   :alt: NLoptLBFGSV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nlopttruncnewtonv1_paths.png
   :width: 100%
   :alt: NLoptTruncNewtonV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptslsqpv1_paths.png
   :width: 100%
   :alt: NLoptSLSQPV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptmmav1_paths.png
   :width: 100%
   :alt: NLoptMMAV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptccsaqv1_paths.png
   :width: 100%
   :alt: NLoptCCSAQV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptsubplexv1_paths.png
   :width: 100%
   :alt: NLoptSubplexV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptcobylaV1_paths.png
   :width: 100%
   :alt: NLoptCOBYLAV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptmlslv1_paths.png
   :width: 100%
   :alt: NLoptMLSLV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptstogov1_paths.png
   :width: 100%
   :alt: NLoptStoGOV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptbobyqav1_paths.png
   :width: 100%
   :alt: NLoptBOBYQAV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptisresv1_paths.png
   :width: 100%
   :alt: NLoptISRESV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nlopteschv1_paths.png
   :width: 100%
   :alt: NLoptESCHV1 minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptshiftvarv1_paths.png
   :width: 100%
   :alt: NLoptShiftVarV1 minimizer paths on a complex Rosenbrock function

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
