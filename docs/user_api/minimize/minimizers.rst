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

Minuit is a longstanding and well proven algorithm of the BFG class implemented in iminuit.
The `iminuit <https://iminuit.readthedocs.io/en/stable/>`_ package is a fast, time-proven
minimizer based on the Minuit2 C++ library. It is an especially robust minimizer that finds the local minimum
quiet reliably for complicated fits. For large fits with hundreds of parameters, other alternatives are generally more performant.
It is however, like all local minimizers, still rather dependent on close enough
initial values.

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

Levenberg-Marquardt minimizer for general non-linear minimization by interpolating between Gauss-Newton and
Gradient descent optimization.

LM minimizes a function by iteratively solving a locally linearized
version of the problem. Using the gradient (g) and the Hessian (H) of
the loss function, the algorithm determines a step (h) that minimizes
the loss function by solving :math:`Hh = g`. This works perfectly in one
step for linear problems, however for non-linear problems it may be
unstable far from the minimum. Thus a scalar damping parameter (L) is
introduced and the Hessian is modified based on this damping.

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

Ipopt is a gradient-based minimizer that performs large scale nonlinear optimization of continuous systems.

This implemenation uses the `IPyOpt wrapper <https://gitlab.com/g-braeunlich/ipyopt>`_

`Ipopt <https://coin-or.github.io/Ipopt/index.html>`_
(Interior Point Optimizer, pronounced "Eye-Pea-Opt") is an open source software package for
large-scale nonlinear optimization. It can be used to solve general nonlinear programming problems
It is written in Fortran and C and is released under the EPL (formerly CPL).
IPOPT implements a primal-dual interior point method, and uses line searches based on
Filter methods (Fletcher and Leyffer).

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

Local, gradient based quasi-Newton algorithm using the BFGS algorithm.

BFGS, named after Broyden, Fletcher, Goldfarb, and Shanno, is a quasi-Newton method
that approximates the Hessian matrix of the loss function using the gradients of the loss function.
It stores an approximation of the inverse Hessian matrix and updates it at each iteration.
For a limited memory version, which doesn't store the full matrix, see L-BFGS-B.

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

Local, gradient based quasi-Newton algorithm using the limited-memory BFGS approximation.

Limited-memory BFGS is an optimization algorithm in the family of quasi-Newton methods
that approximates the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) using a limited amount of
memory (or gradients, controlled by *maxcor*).

L-BFGS borrows ideas from the trust region methods while keeping the L-BFGS update
of the Hessian and line search algorithms.

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

Local minimizer using the modified Powell algorithm.

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

UNSTABLE! Local gradient-free dowhhill simplex-like method with an implicit linear approximation.

COBYLA constructs successive linear approximations of the objective function and constraints via a
simplex of n+1 points (in n dimensions), and optimizes these approximations in a trust region at each step.

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

This minimizer requires the hessian and gradient to be provided by the loss itself.

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

Local, gradient-based quasi-Newton minimizer using the low storage BFGS Hessian approximation.

This is most probably the most popular algorithm for gradient based local minimum searches and also
the underlying algorithm in the
`Minuit <https://www.sciencedirect.com/science/article/abs/pii/0010465575900399>`_ minimizer that is
also available as :class:`~zfit.minimize.Minuit`.

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

Derivative free simplex minimizer using a linear approximation with trust region steps.

COBYLA (Constrained Optimization BY Linear Approximations) constructs successive linear approximations of the
objective function and constraints via a simplex of n+1 points (in n dimensions), and optimizes these
approximations in a trust region at each step.

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

Global minimizer using local optimization by randomly selecting points.

"Multi-Level Single-Linkage" (MLSL) is an algorithm for global optimization by
a sequence of local optimizations from random starting points. MLSL is
distinguished by a "clustering" heuristic that helps it to
avoid repeated searches of the same local optima, and has some
theoretical guarantees of finding all local optima in a finite number of
local minimizations.

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

Derivative-free local minimizer that iteratively constructed quadratic approximation for the loss.

This is an algorithm derived from the BOBYQA subroutine of M. J. D.
Powell, converted to C and modified for the NLopt stopping criteria.
BOBYQA performs derivative-free bound-constrained optimization using an
iteratively constructed quadratic approximation for the objective
function.

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

Improved Stochastic Ranking Evolution Strategy using a mutation rule and differential variation.

The evolution strategy is based on a combination of a mutation rule (with a log-normal step-size update and
exponential smoothing) and differential variation (a Nelder-Mead-like update rule).
The fitness ranking is simply via the objective function for problems without nonlinear constraints,
but when nonlinear constraints are included the stochastic ranking proposed by Runarsson and Yao is employed.

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

Global minimizer using an evolutionary algorithm.

This is a modified Evolutionary Algorithm for global optimization,
developed by Carlos Henrique da Silva Santos's and described in the
following paper and Ph.D thesis.

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
