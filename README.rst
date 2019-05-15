===============================
zfit: scalable pythonic fitting
===============================


.. image:: https://zenodo.org/badge/126311570.svg
   :target: https://zenodo.org/badge/latestdoi/126311570

.. image:: https://img.shields.io/pypi/v/zfit.svg
   :target: https://pypi.python.org/pypi/zfit

.. image:: https://img.shields.io/travis/zfit/zfit.svg
   :target: https://travis-ci.org/zfit/zfit

.. image:: https://coveralls.io/repos/github/zfit/zfit/badge.svg?branch=meta_changes
   :target: https://coveralls.io/github/zfit/zfit?branch=meta_changes

.. image:: https://www.codefactor.io/repository/github/zfit/zfit/badge
   :target: https://www.codefactor.io/repository/github/zfit/zfit
   :alt: CodeFactor

| Quick start with `Interactive Tutorials <https://github.com/zfit/zfit-tutorials>`_
| Read the `Documentation <https://zfit.github.io/zfit>`_ and `API <https://zfit.github.io/zfit/API.html>`_

The zfit package is a model manipulation and fitting library based on `TensorFlow <https://www.tensorflow.org/>`_ and optimised for simple and direct manipulation of probability density functions.
Its main focus is on scalability, parallelisation and user friendly experience.

Detailed documentation, including the API, can be found in https://zfit.github.io/zfit.

It is released as free software following the BSD-3-Clause License.

*N.B.*: zfit is currently in beta stage, so while most core parts are established, some may still be missing and bugs may be encountered.
It is, however, mostly ready for production, and is being used in analyses projects.
If you want to use it for your project and you are not sure if all the needed functionality is there, feel free contact us in our `Gitter channel <https://gitter.im/zfit/zfit>`_.


Why?
----

The basic idea behind zfit is to offer a Python oriented alternative to the very successful RooFit library from the `ROOT <https://root.cern.ch/>`_ data analysis package that can integrate with the other packages that are part if the scientific Python ecosystem.
Contrary to the monolithic approach of ROOT/RooFit, the aim of zfit is to be light and flexible enough to integrate with any state-of-art tools and to allow scalability going to larger datasets.

These core ideas are supported by two basic pillars:

- The skeleton and extension of the code is minimalist, simple and finite:
  the zfit library is exclusively designed for the purpose of model fitting and sampling with no attempt to extend its functionalities to features such as statistical methods or plotting.

- zfit is designed for optimal parallelisation and scalability by making use of TensorFlow as its backend.
  The use of TensorFlow provides crucial features in the context of model fitting like taking care of the parallelisation and analytic derivatives.


Installing
----------

To install zfit, run this command in your terminal:

.. code-block:: console

    $ pip install zfit


For the newest development version, you can install the version from git with

.. code-block:: console

   $ pip install git+https://github.com/zfit/zfit


How to use
----------

While the zfit library provides a simple model fitting and sampling framework for a broad list of applications, we will illustrate its main features by generating, fitting and ploting a Gaussian distribution.

.. code-block:: python

    import zfit

The domain of the PDF is defined by an *observable space*, which is created using the ``zfit.Space`` class:

.. code-block:: python

    obs = zfit.Space('x', limits=(-10, 10))


Using this domain, we can now create a simple Gaussian PDF. To do this, we define its parameters and their limits using the ``zfit.Parameter`` class and we instantiate the PDF from the zfit library:

.. code-block:: python

  # syntax: zfit.Parameter("any_name", value, lower, upper)
    mu    = zfit.Parameter("mu"   , 2.4, -1, 5)
    sigma = zfit.Parameter("sigma", 1.3,  0, 5)
    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)


For simplicity, we create the dataset to be fitted starting from a numpy array, but zfit allows for the use of other sources such as ROOT files:

.. code-block:: python

    mu_true = 0
    sigma_true = 1
    data_np = np.random.normal(mu_true, sigma_true, size=10000)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

Fits are performed in three steps:

1. Creation of a loss function, in our case a negative log-likelihood.
2. Instantiation of our minimiser of choice, in the example the ``MinuitMinimizer``.
3. Minimisation of the loss function.

.. code-block:: python

    # Stage 1: create an unbinned likelihood with the given PDF and dataset
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit)
    minimizer = zfit.minimize.MinuitMinimizer()

    # Stage 3: minimise the given negative log-likelihood
    result = minimizer.minimize(nll)

Errors are calculated with a further function call to avoid running potentially expensive operations if not needed:

.. code-block:: python

    param_errors = result.error()

Once we've performed the fit and obtained the corresponding uncertainties, we can examine the fit results:

.. code-block:: python

    print("Function minimum:", result.fmin)
    print("Converged:", result.converged)
    print("Full minimizer information:", result.info)

    # Information on all the parameters in the fit
    params = result.params
    print(params)

    # Printing information on specific parameters, e.g. mu
    print("mu={}".format(params[mu]['value']))

And that's it!
For more details and information of what you can do with zfit, please see the `documentation page <https://zfit.github.io/zfit>`_.



Contributing
------------

Any idea of how to improve the library? Or interested to write some code?
Contributions are always welcome, please have a look at the `Contributing guide`_.

.. _Contributing guide: CONTRIBUTING.rst

Acknowledgements
----------------

zfit has been developed with support from the University of Zürich and the Swiss National Science Foundation (SNSF) under contracts 168169 and 174182.

The idea of zfit is inspired by the `TensorFlowAnalysis <https://gitlab.cern.ch/poluekt/TensorFlowAnalysis>`_ framework developed by Anton Poluektov using the TensorFlow open source library.

