#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Mapping

import iminuit
import numpy as np

from .. import z
from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter, assign_values
from ..util.cache import GraphCachable
from ..util.deprecation import deprecated_args
from ..util.exception import MaximumIterationReached
from .baseminimizer import BaseMinimizer, minimize_supports, print_minimization_status
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import EDM, ConvergenceCriterion


class Minuit(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "Minuit"

    @deprecated_args(None, "Use `options` instead.", "minimizer_options")
    @deprecated_args(None, "Use `maxiter` instead.", "ncall")
    @deprecated_args(None, "Use `mode` instead.", "minimize_strategy")
    @deprecated_args(None, "Use `gradient` instead.", "minuit_grad")
    @deprecated_args(None, "Use `gradient` instead.", "use_minuit_grad")
    def __init__(
        self,
        tol: float | None = None,
        mode: int | None = None,
        gradient: bool | str | None = None,
        verbosity: int | None = None,
        options: Mapping[str, object] | None = None,
        maxiter: int | None = None,
        criterion: ConvergenceCriterion | None = None,
        strategy: ZfitStrategy | None = None,
        name: str | None = None,
        # legacy arguments
        use_minuit_grad: bool | None = None,
        minuit_grad=None,
        minimize_strategy=None,
        ncall=None,
        minimizer_options=None,
    ):
        """Minuit is a longstanding and well proven algorithm of the L-BFGS-B class implemented in iminuit.

        The `iminuit <https://iminuit.readthedocs.io/en/stable/>`_ package is a fast, time-proven
        minimizer based on the Minuit2 C++ library; the latter is
        maintained by CERN's ROOT team. It is an especially robust minimizer that finds the global minimum
        quiet reliably. It is however, like all local minimizers, still rather dependent on close enough
        initial values.

        Args:
            tol:  |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            mode: A number used by minuit to define the internal minimization strategy, either 0, 1 or 2.
                As `explained in the iminuit docs <https://iminuit.readthedocs.io/en/stable/faq.html#what-happens-when-i-change-the-strategy>`_
                , they mean:
                - 0 The fastest and the number of function calls required to minimise
                    scales linearly with the number of fitted parameters. The Hesse matrix is not computed during the
                    minimisation (only an approximation that is continuously updated).
                    When the number of fitted parameters > 10, you should prefer this strategy.
                - 1 (default with Minuit gradient) medium in speed. The number of function calls required
                    scales quadratically with the number of fitted parameters. The different scales comes from the fact
                     that the Hesse matrix is explicitly computed in a Newton step,
                     if Minuit detects significant correlations between parameters.
                - 2 same quadratic scaling as strategy 1 but is even slower. The Hesse matrix is
                    always explicitly computed in each Newton step.
            gradient: If True, iminuit uses its internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit. If False or ``'zfit'``, the latter
                is used. For smaller datasets with less stable losses, the internal Minuit gradient often performs
                better while the zfit provided gradient improves the convergence rate for larger (10'000+) datasets.
            verbosity: |@doc:minimizer.verbosity| Verbosity of the minimizer. Has to be between 0 and 10.
              The verbosity has the meaning:

               - a value of 0 means quiet and no output
               - above 0 up to 5, information that is good to know but without
                 flooding the user, corresponding to a "INFO" level.
               - A value above 5 starts printing out considerably more and
                 is used more for debugging purposes.
               - Setting the verbosity to 10 will print out every
                 evaluation of the loss function and gradient.

               Some minimizers offer additional output which is also
               distributed as above but may duplicate certain printed values. |@docend:minimizer.verbosity| This
                also changes the iminuit internal verbosity at around 7.
            options: Additional options that will be directly passsed into :meth:`~iminuitMinuit.migrad`
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|


            use_minuit_grad: deprecated, legacy.
            minuit_grad: deprecated, legacy.
            minimize_strategy: deprecated, legacy.
            ncall: deprecated, legacy.
            minimizer_options: deprecated, legacy.
        """
        # legacy
        if isinstance(mode, float) or isinstance(tol, int):
            msg = "mode has to be int, tol a float. The API changed, make sure you use the" " right parameters."
            raise TypeError(msg)
        if minimizer_options is not None:
            options = minimizer_options
        if ncall is not None:
            maxiter = ncall
        if minimize_strategy is not None:
            mode = minimize_strategy
        use_grad_legacy = use_minuit_grad if use_minuit_grad is not None else minuit_grad
        if use_grad_legacy is not None:
            gradient = use_grad_legacy
        # end legacy

        if gradient == "zfit":
            gradient = False
        gradient = True if gradient is None else gradient

        self._internal_maxiter = 20

        options = {} if options is None else options
        options["ncall"] = 0 if maxiter is None else maxiter
        if mode is None:
            mode = 1
        if mode not in range(3):
            msg = f"mode has to be 0, 1 or 2, not {mode}."
            raise ValueError(msg)
        options["strategy"] = mode

        super().__init__(
            name=name,
            strategy=strategy,
            tol=tol,
            verbosity=verbosity,
            criterion=criterion,
            maxiter=1e20,
            minimizer_options=options,
        )
        self._minuit_minimizer = None
        self._use_tfgrad_internal = not gradient
        self.minuit_grad = gradient

    # TODO 0.7: legacy, remove `_use_tfgrad`
    @property
    def _use_tfgrad(self):
        from zfit.exception import BreakingAPIChangeError

        msg = "This property is not available anymore. Use `gradient` instead."
        raise BreakingAPIChangeError(msg)

    @minimize_supports(init=True)
    def _minimize(self, loss: ZfitLoss, params: list[Parameter], init):
        if init:
            assign_values(params=params, values=init)
        criterion = self.create_criterion(loss, params)

        minimizer, minimize_options, evaluator = self._make_minuit(loss, params, init)

        self._minuit_minimizer = minimizer

        valid = False
        message = ""
        maxiter_reached = False
        for i in range(self._internal_maxiter):
            # perform minimization
            try:
                minimizer = minimizer.migrad(**minimize_options)
            except MaximumIterationReached as error:
                if minimizer is None:  # it didn't even run once
                    msg = (
                        "Maximum iteration reached on first wrapped minimizer call. This"
                        "is likely to a too low number of maximum iterations (currently"
                        f" {evaluator.maxiter}) or wrong internal tolerances, in which"
                        f" case: please fill an issue on github."
                    )
                    raise MaximumIterationReached(msg) from error
                maxiter_reached = True
                message = "Maxiter reached"
            else:
                if evaluator.maxiter is not None:
                    maxiter_reached = evaluator.niter > evaluator.maxiter
            if type(criterion) is EDM:  # use iminuits edm
                criterion.last_value = minimizer.fmin.edm
                converged = not minimizer.fmin.is_above_max_edm
            else:
                fitresult = FitResult.from_minuit(
                    loss=loss,
                    params=params,
                    minuit=minimizer,
                    minimizer=self,
                    valid=valid,
                    message=message,
                )
                converged = criterion.converged(fitresult)

            if self.verbosity > 5:
                internal_tol = {"edm_minuit": minimizer.fmin.edm}

                print_minimization_status(
                    converged=converged,
                    criterion=criterion,
                    evaluator=evaluator,
                    i=i,
                    fminopt=minimizer.fval,
                    internal_tol=internal_tol,
                )

            if converged or maxiter_reached:
                assign_values(params, z.convert_to_tensor(minimizer.values))  # make sure it's at the right value
                if not maxiter_reached:
                    valid = True
                break

        return FitResult.from_minuit(
            loss=loss,
            params=params,
            criterion=criterion,
            minuit=minimizer,
            minimizer=self.copy(),
            valid=valid,
            message=message,
        )

    def _make_minuit(self, loss, params, init):
        evaluator = self.create_evaluator(loss, params)

        # create options
        minimizer_options = self.minimizer_options.copy()
        minimize_options = {}
        precision = minimizer_options.pop("precision", None)
        minimize_options["ncall"] = minimizer_options.pop("ncall")
        minimizer_init = {}
        if "errordef" in minimizer_options:
            msg = "errordef cannot be specified for Minuit as this is already defined in the Loss."
            raise ValueError(msg)
        loss_errordef = loss.errordef
        if not isinstance(loss_errordef, (float, int)):
            msg = "errordef has to be a float"
            raise ValueError(msg)
        minimizer_init["errordef"] = loss_errordef
        minimizer_init["pedantic"] = minimizer_options.pop("pedantic", False)
        minimizer_setter = {}
        minimizer_setter["strategy"] = minimizer_options.pop("strategy")
        if self.verbosity > 8:
            minuit_verbosity = 3
        elif self.verbosity > 6:
            minuit_verbosity = 2
        elif self.verbosity > 1:
            minuit_verbosity = 1
        else:
            minuit_verbosity = 0
        if minimizer_options:
            msg = f"The following options are not (yet) supported: {minimizer_options}"
            raise ValueError(msg)
        init_values = np.array(params)

        # create Minuit compatible names
        params_name = [param.name for param in params]
        # TODO 0.7: legacy, remove `_use_tfgrad`
        grad_func = evaluator.gradient if self._use_tfgrad_internal or not self.minuit_grad else None
        minimizer = iminuit.Minuit(
            evaluator.value,
            init_values,
            grad=grad_func,
            name=params_name,
        )
        minimizer.precision = precision
        approx_stepsizes = {}
        # get possible initial step size from previous minimizer
        if init:
            approx_stepsizes = init.hesse(params=params, method="approx", name="approx")

        empty_dict = {}
        for param in params:
            stepsize = approx_stepsizes.get(param, empty_dict).get("error")
            if stepsize is None and param.has_stepsize:
                stepsize = param.stepsize
            if stepsize is not None:
                minimizer.errors[param.name] = stepsize
        # set limits
        for param in params:
            if param.has_limits:
                minimizer.limits[param.name] = (param.lower, param.upper)
        # set options
        minimizer.errordef = loss.errordef
        minimizer.print_level = minuit_verbosity
        strategy = minimizer_setter.pop("strategy")
        minimizer.strategy = strategy
        minimizer.tol = (
            self.tol / 0.002 / loss.errordef  # iminuit multiplies by default with 0.002
        )  # to account for the loss
        assert (
            not minimizer_setter
        ), f"minimizer_setter is not empty, bug. Please report. minimizer_setter: {minimizer_setter}"
        return minimizer, minimize_options, evaluator

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer
