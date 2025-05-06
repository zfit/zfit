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


The following visualization shows how different minimizers perform on a complex version of the Rosenbrock function.
Each minimizer starts from 5 different initial points (marked with symbols) and follows a path to the minimum.
The contour plot represents the function values, with darker colors indicating lower values.
For each minimizer, the plot shows the number of function evaluations, gradient calculations, and execution time.


Minuit
:::::::

.. image:: ../../images/_generated/minimizers/minuit_paths.gif
   :width: 100%
   :alt: Minuit minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/minuit_paths_static.png
   :width: 100%
   :alt: Minuit minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.Minuit

Levenberg-Marquardt
:::::::::::::::::::::

.. image:: ../../images/_generated/minimizers/levenbergmarquardt_paths.gif
   :width: 100%
   :alt: LevenbergMarquardt minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/levenbergmarquardt_paths_static.png
   :width: 100%
   :alt: LevenbergMarquardt minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.LevenbergMarquardt


Ipyopt
:::::::

.. image:: ../../images/_generated/minimizers/ipyopt_paths.gif
   :width: 100%
   :alt: Ipyopt minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/ipyopt_paths_static.png
   :width: 100%
   :alt: Ipyopt minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.Ipyopt


Scipy
::::::

The following visualizations show how different Scipy minimizers perform on the complex Rosenbrock function.

BFGS
------------------------

.. image:: ../../images/_generated/minimizers/scipybfgs_paths.gif
   :width: 100%
   :alt: ScipyBFGS minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipybfgs_paths_static.png
   :width: 100%
   :alt: ScipyBFGS minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyBFGS

LBFGSB
------------------------

.. image:: ../../images/_generated/minimizers/scipylbfgsb_paths.gif
   :width: 100%
   :alt: ScipyLBFGSB minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipylbfgsb_paths_static.png
   :width: 100%
   :alt: ScipyLBFGSB minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyLBFGSB

TrustConstr
------------------------

.. image:: ../../images/_generated/minimizers/scipytrustconstr_paths.gif
   :width: 100%
   :alt: ScipyTrustConstr minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustconstr_paths_static.png
   :width: 100%
   :alt: ScipyTrustConstr minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyTrustConstr

Powell
------------------------

.. image:: ../../images/_generated/minimizers/scipypowell_paths.gif
   :width: 100%
   :alt: ScipyPowell minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipypowell_paths_static.png
   :width: 100%
   :alt: ScipyPowell minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyPowell

SLSQP
------------------------

.. image:: ../../images/_generated/minimizers/scipyslsqp_paths.gif
   :width: 100%
   :alt: ScipySLSQP minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipyslsqp_paths_static.png
   :width: 100%
   :alt: ScipySLSQP minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipySLSQP

TruncNC
------------------------

.. image:: ../../images/_generated/minimizers/scipytruncnc_paths.gif
   :width: 100%
   :alt: ScipyTruncNC minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytruncnc_paths_static.png
   :width: 100%
   :alt: ScipyTruncNC minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyTruncNC

COBYLA
------------------------

.. image:: ../../images/_generated/minimizers/scipycobyla_paths.gif
   :width: 100%
   :alt: ScipyCOBYLA minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipycobyla_paths_static.png
   :width: 100%
   :alt: ScipyCOBYLA minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyCOBYLA

TrustNCG
------------------------

.. image:: ../../images/_generated/minimizers/scipytrustncg_paths.gif
   :width: 100%
   :alt: ScipyTrustNCG minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustncg_paths_static.png
   :width: 100%
   :alt: ScipyTrustNCG minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyTrustNCG

Dogleg
------------------------

.. image:: ../../images/_generated/minimizers/scipydogleg_paths.gif
   :width: 100%
   :alt: ScipyDogleg minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipydogleg_paths_static.png
   :width: 100%
   :alt: ScipyDogleg minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyDogleg

ScipyTrustKrylov
------------------------

.. image:: ../../images/_generated/minimizers/scipytrustkrylov_paths.gif
   :width: 100%
   :alt: ScipyTrustKrylov minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipytrustkrylov_paths_static.png
   :width: 100%
   :alt: ScipyTrustKrylov minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyTrustKrylov

NewtonCG
--------

.. image:: ../../images/_generated/minimizers/scipynewtoncg_paths.gif
   :width: 100%
   :alt: ScipyNewtonCG minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/scipynewtoncg_paths_static.png
   :width: 100%
   :alt: ScipyNewtonCG minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.ScipyNewtonCG





NLopt
::::::

The following visualizations show how different NLopt minimizers perform on the complex Rosenbrock function:

LBFGS
------

.. image:: ../../images/_generated/minimizers/nloptlbfgs_paths.gif
   :width: 100%
   :alt: NLoptLBFGS minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptlbfgs_paths_static.png
   :width: 100%
   :alt: NLoptLBFGS minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptLBFGS

Truncated Newton
----------------

.. image:: ../../images/_generated/minimizers/nlopttruncnewton_paths.gif
   :width: 100%
   :alt: NLoptTruncNewton minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nlopttruncnewton_paths_static.png
   :width: 100%
   :alt: NLoptTruncNewton minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptTruncNewton

SLSQP
-----

.. image:: ../../images/_generated/minimizers/nloptslsqp_paths.gif
   :width: 100%
   :alt: NLoptSLSQP minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptslsqp_paths_static.png
   :width: 100%
   :alt: NLoptSLSQP minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptSLSQP

MMA
---

.. image:: ../../images/_generated/minimizers/nloptmma_paths.gif
   :width: 100%
   :alt: NLoptMMA minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptmma_paths_static.png
   :width: 100%
   :alt: NLoptMMA minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptMMA

CCSAQ
-----

.. image:: ../../images/_generated/minimizers/nloptccsaq_paths.gif
   :width: 100%
   :alt: NLoptCCSAQ minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptccsaq_paths_static.png
   :width: 100%
   :alt: NLoptCCSAQ minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptCCSAQ

Subplex
-------

.. image:: ../../images/_generated/minimizers/nloptsubplex_paths.gif
   :width: 100%
   :alt: NLoptSubplex minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptsubplex_paths_static.png
   :width: 100%
   :alt: NLoptSubplex minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptSubplex

COBYLA
------

.. image:: ../../images/_generated/minimizers/nloptcobyla_paths.gif
   :width: 100%
   :alt: NLoptCOBYLA minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptcobyla_paths_static.png
   :width: 100%
   :alt: NLoptCOBYLA minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptCOBYLA

MLSL
----

.. image:: ../../images/_generated/minimizers/nloptmlsl_paths.gif
   :width: 100%
   :alt: NLoptMLSL minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptmlsl_paths_static.png
   :width: 100%
   :alt: NLoptMLSL minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptMLSL

StoGO
-----

.. image:: ../../images/_generated/minimizers/nloptstogo_paths.gif
   :width: 100%
   :alt: NLoptStoGO minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptstogo_paths_static.png
   :width: 100%
   :alt: NLoptStoGO minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptStoGO

BOBYQA
------

.. image:: ../../images/_generated/minimizers/nloptbobyqa_paths.gif
   :width: 100%
   :alt: NLoptBOBYQA minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptbobyqa_paths_static.png
   :width: 100%
   :alt: NLoptBOBYQA minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptBOBYQA

ISRES
-----

.. image:: ../../images/_generated/minimizers/nloptisres_paths.gif
   :width: 100%
   :alt: NLoptISRES minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptisres_paths_static.png
   :width: 100%
   :alt: NLoptISRES minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptISRES

ESCH
----

.. image:: ../../images/_generated/minimizers/nloptesch_paths.gif
   :width: 100%
   :alt: NLoptESCH minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptesch_paths_static.png
   :width: 100%
   :alt: NLoptESCH minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptESCH

ShiftVar
--------

.. image:: ../../images/_generated/minimizers/nloptshiftvar_paths.gif
   :width: 100%
   :alt: NLoptShiftVar minimizer paths on a complex Rosenbrock function

.. image:: ../../images/_generated/minimizers/nloptshiftvar_paths_static.png
   :width: 100%
   :alt: NLoptShiftVar minimizer paths on a complex Rosenbrock function

.. autosummary::

    zfit.minimize.NLoptShiftVar

All minimizers
::::::::::::::::

.. autosummary::
    :toctree: _generated/minimizers

    zfit.minimize.Minuit
    zfit.minimize.LevenbergMarquardt
    zfit.minimize.Ipyopt
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
    zfit.minimize.NLoptLBFGS
    zfit.minimize.NLoptTruncNewton
    zfit.minimize.NLoptSLSQP
    zfit.minimize.NLoptMMA
    zfit.minimize.NLoptCCSAQ
    zfit.minimize.NLoptSubplex
    zfit.minimize.NLoptCOBYLA
    zfit.minimize.NLoptMLSL
    zfit.minimize.NLoptStoGO
    zfit.minimize.NLoptBOBYQA
    zfit.minimize.NLoptISRES
    zfit.minimize.NLoptESCH
    zfit.minimize.NLoptShiftVar
