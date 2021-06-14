Minimization
============

Minimizers are the second last key element in the API framework of zfit.
In particular, these are connected to the loss function that they minimize.

zfit minimizers are stateless and have two main components:

- the creation of a minimizer instance which has some common arguments such as ``tol``, ``verbosity`` or ``maxiter``
  and many minimizer specific arguments such as ``gradient``, ``max_trust_radius`` that only a few or one
  minimizer takes. This instance will be stateless and anything (inherently) stateful, such as the convergence
  criterion, are created newly during each minimization process.
  In this sense, a zfit minimizer is a way of "storing a configuration".
- the actual minimization which is done through the :meth:`~zfit.minimize.BaseMinimizer.minimize` method, which
  takes a loss (or a callable), optionally the parameters and optionally a previous result to start from. This method
  looks exactly the same for all algorithms.

This makes minimizers interchangeable as they all are invoked in the same way and return a
:class:`~zfit.result.FitResult`, which has the same structure for all minimizers.

The zfit library is designed such that it is simple to introduce new sets of minimizers.

Minimization
-------------------

There are multiple minimizers currently included in the package: :class:`~zfit.minimize.IpyoptV1`,
:class:`~zfit.minimize.Minuit`, the SciPy optimizers (such as :class:`~zfit.minimize.ScipyLBFGSBV1`) and the
NLopt library (such as :class:`~zfit.minimize.NLoptLBFGSV1`). Also TensorFlow minimizers are included, however due
to the different nature of problem that they usually intend to solve, their performance is often inferior.

.. code-block:: pycon

    >>> # Minuit minimizer
    >>> minimizer_minuit = zfit.minimize.Minuit()
    >>> # Ipyopt minimizer
    >>> minimizer_ipyopt = zfit.minimize.IpyoptV1()
    >>> # One of the NLopt minimizer
    >>> minimizer_nloptbfgs = zfit.minimize.NLoptLBFGSV1()
    >>> # One of the SciPy minimizer
    >>> minimizer_scipybfgs = zfit.minimize.ScipyLBFGSBV1()

.. note:: Why the "V1" at the end of the name?

    This minimizers and their API have been introduced recently. Due to their stochastic nature, it is hard
    to reliably assess their performance *without large scale testing by the community*. So there will be
    improved versions, called V2 etc, which can be tested easily against the V1 in order to have a direct
    comparison. At some point later, there will be only one minimizer version, the one without any V.


Any of these minimizers can then be used to minimize the loss function we created
in the :ref:`previous section <loss-section>`, or a pure Python function

.. code-block:: pycon

    >>> result = minimizer_minuit.minimize(loss=my_loss)

The choice of which parameters of your model should be floating in the fit can also be made at this stage

.. code-block:: pycon

    >>> # In the case of a Gaussian (e.g.)
    >>> result = minimizer_minuit.minimize(loss=my_loss, params=[mu, sigma])

**Only** the parameters given in ``params`` are floated in the optimisation process.
If this argument is not provided or ``params=None``, all the floating parameters in the loss function are
floated in the minimization process.

The third argument to minimize can be a :class:`~zfit.result.FitResult` that initializes the minimization. This can be
used to possibly chain minimizations and for example first search with a global minimizer at a high tolerance and then
refine with a local minimizer.

.. code-block:: pycon

    >>> result_refined = minimizer_ipyopt.minimize(loss=my_loss, params=[mu, sigma], init=result)


The result of the fit is returned as a :py:class:`~zfit.minimizers.fitresult.FitResult` object,
which provides access the minimiser state.
zfit separates the minimisation of the loss function with respect to the error calculation
in order to give the freedom of calculating this error whenever needed.

The ``result`` object contains all the information about the minimization result:

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

More on the result and how to get an estimate of the uncertainty is described in
the :ref:`next section <result-section>`.


Creating your own minimizer
----------------------------

Adding new minimizers is well possible in zfit as there are convenient base classes offered that take most of the heavy
lifting.

While this is a feature of zfit that can be fully used, it will not be as stable as the simple usage of a minimizer
until the 1.0 release.


A wrapper for TensorFlow optimisers is also available to allow to easily integrate new ideas in the framework.
For instance, the Adam minimizer could have been initialised by

.. code-block:: pycon

    >>> # Adam's TensorFlor optimiser using a wrapper
    >>> minimizer_wrapper = zfit.minimize.WrapOptimizer(tf.keras.optimizer.Adam())
