.. _result-section:


Result
============


As seen before, the ``result`` object contains all the information about the minimization result:

.. code-block:: pycon

    >>> print("Function minimum:", result.fmin)
    Function minimum: 14170.396450111948
    >>> print("Converged:", result.converged)
    Converged: True
    >>> print("Valid:", result.valid)
    Valid: True
    >>> print("Full minimizer information:", result)



.. code-block:: pycon

    >>> # Information on all the parameters in the fit
    >>> params = result.params

    >>> # Printing information on specific parameters, e.g. mu
    >>> print("mu={}".format(params[mu]['value']))
    mu=0.012464509810750313

Estimating uncertainties
----------------------------

In order to get an estimate for the uncertainty of the parameters after the minimization, the FitResult can be used.

.. code-block:: pycon

    >>> param_hesse = result.hesse()
    >>> print(param_hesse)

This will print out the uncertainties of the parameter using a Hessian approximation at the minimum of the loss.
While the approximation is fast and often good enough, it is symmetric and does maybe not describe the uncertainty
well.

The :py:func:`~zfit.minimizers.fitresult.FitResult.errors` method can be used to perform the CPU-intensive
error calculation.
It returns two objects, the first are the parameter errors and the second is a new ``FitResult`` *in case a new
minimum was found during the profiling*; this will also render the original result invalid as can
be checked with ``result.valid``.

.. code-block:: pycon

    >>> param_errors, _ = result.errors()
    >>> print(param_errors)

This will print out the uncertainties of the parameter using a profiling method (like :meth:`~iminuit.Minuit.minos`
does)
