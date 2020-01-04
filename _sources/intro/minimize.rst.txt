Minimization
============

Minimizer objects are the last key element in the API framework of zfit.
In particular, these are connected to the loss function and have an internal state that can be queried at any moment.

The zfit library is designed such that it is trivial to introduce new sets of minimizers.
The only requirement in its initialisation is that a loss function **must** be given.
Additionally, the parameters to be minimize, the tolerance, its name, as well as any other argument needed to configure the particular algorithm **may** be given.

Baseline minimizers
-------------------

There are three minimizers currently included in the package: ``Minuit``, ``Scipy`` and ``Adam`` TensorFlow optimiser.
Let's show how these can be initialised:

.. code-block:: pycon

    >>> # Minuit minimizer
    >>> minimizer_minuit = zfit.minimize.Minuit()
    >>> # Scipy minimizer
    >>> minimizer_scipy = zfit.minimize.Scipy()
    >>> # Adam's Tensorflow minimizer
    >>> minimizer_adam = zfit.minimize.Adam()

A wrapper for TensorFlow optimisers is also available to allow to easily integrate new ideas in the framework.
For instance, the Adam minimizer could have been initialised by

.. code-block:: pycon

    >>> # Adam's TensorFlor optimiser using a wrapper
    >>> minimizer_wrapper = zfit.minimize.WrapOptimizer(tf.keras.optimizer.Adam())

Any of these minimizers can then be used to minimize the loss function we created in :ref:`previous section <data-section>`, e.g.

.. code-block:: pycon

    >>> result = minimizer_minuit.minimize(loss=my_loss)

The choice of which parameters of your model should be floating in the fit can also be made at this stage

.. code-block:: pycon

    >>> # In the case of a Gaussian (e.g.)
    >>> result = minimizer_minuit.minimize(loss=my_loss, params=[mu, sigma])

**Only** the parameters given in ``params`` are floated in the optimisation process.
If this argument is not provided or ``params=None``, all the floating parameters in the loss function are floated in the minimization process.

The result of the fit is return as a :py:class:`~zfit.minimizers.fitresult.FitResult` object, which provides access the minimiser state.
zfit separates the minimisation of the loss function with respect to the error calculation in order to give the freedom of calculating this error whenever needed.
The :py:func:`~zfit.minimizers.fitresult.FitResult.error` method can be used to perform the CPU-intensive error calculation.

.. code-block:: pycon

    >>> param_errors = result.error()
    >>> for var, errors in param_errors.items():
    ...   print('{}: ^{{+{}}}_{{-{}}}'.format(var.name, errors['upper'], errors['lower']))
    mu: ^{+0.00998104141841555}_{--0.009981515893414316}
    sigma: ^{+0.007099472590970696}_{--0.0070162654764939734}


The ``result`` object also provides access the minimiser state:

.. code-block:: pycon

    >>> print("Function minimum:", result.fmin)
    Function minimum: 14170.396450111948
    >>> print("Converged:", result.converged)
    Converged: True
    >>> print("Full minimizer information:", result.info)
    Full minimizer information: {'n_eval': 56, 'original': {'fval': 14170.396450111948, 'edm': 2.8519671693442587e-10,
    'nfcn': 56, 'up': 0.5, 'is_valid': True, 'has_valid_parameters': True, 'has_accurate_covar': True, 'has_posdef_covar': True,
    'has_made_posdef_covar': False, 'hesse_failed': False, 'has_covariance': True, 'is_above_max_edm': False, 'has_reached_call_limit': False}}

and the fitted parameters

.. code-block:: pycon

    >>> # Information on all the parameters in the fit
    >>> params = result.params

    >>> # Printing information on specific parameters, e.g. mu
    >>> print("mu={}".format(params[mu]['value']))
    mu=0.012464509810750313


