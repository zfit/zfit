#  Copyright (c) 2021 zfit
from typing import Optional, Dict

import nlopt
import numpy as np
import tensorflow as tf

from .baseminimizer import BaseMinimizer, ZfitStrategy, print_gradients, print_params
from .fitresult import FitResult
from ..core.parameter import set_values
from ..settings import run


class NLopt(BaseMinimizer):
    def __init__(self, algorithm: int = nlopt.LD_LBFGS, tolerance: Optional[float] = None,
                 strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5, name: Optional[str] = None,
                 maxiter: Optional[int] = None, minimizer_options: Optional[Dict[str, object]] = None):
        """NLopt contains multiple different optimization algorithms.py

        `NLopt <https://nlopt.readthedocs.io/en/latest/>`_ is a free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as original implementations of various other
        algorithms.

        Args:
            algorithm: Define which algorithm to be used. These are taken from `nlopt.ALGORITHM` (where `ALGORITHM` is
                the actual algorithm). A comprehensive list and description of all implemented algorithms is
                available `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.
                The wrapper is optimized for Local gradient-based optimization and may breaks with
                others. However, please post a feature request in case other algorithms are requested.

                The naming of the algorithm starts with either L/G (Local/Global)
                 and N/D (derivative-free/derivative-based).

                Local optimizer ptions include (but not only)

                Derivative free:
                - LN_NELDERMEAD: The Nelder Mead Simplex algorithm, which seems to perform not so well, but can be used
                  to narrow down a minimum.
                - LN_SBPLX: SubPlex is an improved version of the Simplex algorithms and usually performs better.

                With derivative:
                - LD_MMA: Method of Moving Asymptotes, an improved CCSA
                  ("conservative convex separable approximation") variant of the original MMA algorithm
                - LD_SLSQP: this is a sequential quadratic programming (SQP) algorithm for (nonlinearly constrained)
                  gradient-based optimization
                - LD_LBFGS: version of the famous low-storage BFGS algorithm, an approximate Newton method. The same as
                  the Minuit algorithm is built on.
                - LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, LD_TNEWTON_RESTART, LD_TNEWTON: a preconditioned
                  inexact truncated Newton algorithm. Multiple variations, with and without preconditioning and/or
                  restart are provided.
                - LD_VAR1, LD_VAR2: a shifted limited-memory variable-metric algorithm, either using a rank 1 or rank 2
                  method.

            tolerance (Union[float, None]):
            strategy (Union[None, None]):
            verbosity (int):
            name (Union[None, None]):
            maxiter (Union[None, None]):
            minimizer_options (Union[None, None]):
        """
        self.algorithm = algorithm
        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_options=minimizer_options,
                         strategy=strategy, maxiter=maxiter)

    def _minimize(self, loss, params):
        minimizer = nlopt.opt(self.algorithm, len(params))

        current_loss = None
        current_grad = None
        nan_counter = 0
        n_eval = 0

        init_val = np.array(run(params))

        def func(values):
            nonlocal current_loss, nan_counter, n_eval
            n_eval += 1
            self._update_params(params=params, values=values)
            do_print = self.verbosity > 8

            is_nan = False

            try:
                loss_evaluated = loss.value().numpy()
            except tf.errors.InvalidArgumentError:
                is_nan = True
                loss_evaluated = "invalid, error occured"
            except:
                loss_evaluated = "invalid, error occured"
                raise
            finally:
                if do_print:
                    print_params(params, values, loss_evaluated)
            is_nan = is_nan or np.isnan(loss_evaluated)
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values['loss'] = loss_evaluated
                info_values['old_loss'] = current_loss
                info_values['nan_counter'] = nan_counter
                loss_evaluated = self.strategy.minimize_nan(loss=loss, params=params, minimizer=self,
                                                            values=info_values)
            else:
                nan_counter = 0
                current_loss = loss_evaluated
            return loss_evaluated

        def grad_value_func(values):
            nonlocal current_loss, current_grad, nan_counter, n_eval
            n_eval += 1
            self._update_params(params=params, values=values)
            do_print = self.verbosity > 8
            is_nan = False

            try:
                loss_value, gradients = loss.value_gradients(params=params)
                loss_value = loss_value.numpy()
                gradients_values = [float(g.numpy()) for g in gradients]
            except tf.errors.InvalidArgumentError:
                is_nan = True
                loss_value = "invalid, error occured"
                gradients_values = ["invalid"] * len(params)
            except:
                gradients_values = ["invalid"] * len(params)
                raise
            finally:
                if do_print:
                    try:
                        print_gradients(params, values, gradients_values, loss=loss_value)
                    except:
                        print("Cannot print loss value or gradient values.")

            is_nan = is_nan or any(np.isnan(gradients_values)) or np.isnan(loss_value)
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values['loss'] = loss_value
                info_values['grad'] = loss_value
                info_values['old_loss'] = current_loss
                info_values['old_grad'] = current_grad
                info_values['nan_counter'] = nan_counter
                # but loss value not needed here
                loss_value= self.strategy.minimize_nan(loss=loss, params=params, minimizer=self,
                                                        values=info_values)
                gradients_values = np.zeros_like(gradients_values)
            else:
                nan_counter = 0
                current_loss = loss_value
                gradients_values = np.zeros_like(gradients_values)
            return loss_value, gradients_values

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.maxiter)
        minimizer.set_ftol_abs(self.tolerance)

        for name, value in self.minimizer_options:
            minimizer.set_param(name, value)

        values = minimizer.optimize(init_val)
        set_values(params, values)

        param_dict = {p: v for p, v in zip(params, values)}
        edm = -999
        fmin = minimizer.last_optimum_value()
        status = minimizer.last_optimize_result()
        converged = 1 <= status <= 4
        messages = {
            1: "NLOPT_SUCCESS",
            2: "NLOPT_STOPVAL_REACHED",
            3: "NLOPT_FTOL_REACHED",
            4: "NLOPT_XTOL_REACHED",
            5: "NLOPT_MAXEVAL_REACHED",
            6: "NLOPT_MAXTIME_REACHED",
            -1: "NLOPT_FAILURE",
            -2: "NLOPT_INVALID_ARGS",
            -3: "NLOPT_OUT_OF_MEMORY",
            -4: "NLOPT_ROUNDOFF_LIMITED",
            -5: "NLOPT_FORCED_STOP",
        }
        message = messages[status]
        info = {'n_eval': n_eval,
                'message': message,
                'original': status,
                'status': status}

        return FitResult(
            params=param_dict,
            edm=edm,
            fmin=fmin,
            status=status,
            converged=converged,
            info=info,
            loss=loss,
            minimizer=self.copy()
        )
