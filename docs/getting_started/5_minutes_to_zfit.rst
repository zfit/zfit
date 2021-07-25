.. _5-minutes-to-zfit:

=================
5 minutes to zfit
=================

The zfit library provides a simple model fitting and sampling framework for a broad list of applications.
This section is designed to give an overview of the main concepts and features in the context of likelihood fits in
a *crash course* manner. The simplest example is to generate, fit and plot a Gaussian distribution.

The first step is to import ``zfit`` and to verify that the installation has been done successfully:

.. jupyter-execute::
    :hide-output:
    :hide-code:

    import os
    os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"
    import numpy as np


.. jupyter-execute::

    import tensorflow as tf
    import zfit
    from zfit import z  # math backend of zfit

.. thebe-button:: Run this interactively


Since we want to generate/fit a Gaussian within a given range, the domain of the PDF is defined by
an *observable space*. This can be created using the :py:class:`~zfit.Space` class

.. jupyter-execute::

    obs = zfit.Space('x', limits=(-10, 10))

The best interpretation of the observable at this stage is that it defines the name and range of the observable axis.

Using this domain, we can now create a simple Gaussian PDF.
The most common PDFs are already pre-defined within the :py:mod:`~zfit.pdf` module, including a simple Gaussian.
First, we have to define the parameters of the PDF and their limits using the :py:class:`~zfit.Parameter` class:

.. jupyter-execute::

    mu = zfit.Parameter("mu", 2.4, -1, 5)
    sigma = zfit.Parameter("sigma", 1.3,  0, 5)

With these parameters we can instantiate the Gaussian PDF from the library

.. jupyter-execute::

    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

It is recommended to pass the arguments of the PDF as keyword arguments.

The next stage is to create a dataset to be fitted. There are several ways of producing this within the
zfit framework (see the :ref:`Data <data-section>` section). In this case, for simplicity we simply produce
it using numpy and the :func:`Data.from_numpy <zfit.Data.from_numpy>` method:

.. jupyter-execute::

    data_np = np.random.normal(0, 1, size=10000)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

Now we have all the ingredients in order to perform a maximum likelihood fit.
Conceptually this corresponds to three basic steps:

1. create a loss function, in our case a negative log-likelihood :math:`\log\mathcal{L}`;
2. instantiate our choice of minimiser;
3. and minimise the log-likelihood.

.. jupyter-execute::

    # Stage 1: create an unbinned likelihood with the given PDF and dataset
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit minimizer)
    minimizer = zfit.minimize.Minuit()

    # Stage 3: minimise the given negative likelihood
    result = minimizer.minimize(nll)

This corresponds to the most basic example where the negative likelihood is defined within the pre-determined
observable range and all the parameters in the PDF are floated in the fit. It is often the case that we want to
only vary a given set of parameters. In this case it is necessary to specify which are the parameters to be floated
(so all the remaining ones are fixed to their initial values).

Also note that we can now do various things with the pdf such as plotting the fitting result
with the model gaussian without extracting the loss
minimizing parameters from ``result``. This is possible because parameters are mutable. This means that the
minimizer can directly manipulate the value of the floating parameter. So when you call the ``minimizer.minimize()``
method the value of ``mu`` changes during the optimisation. ``gauss.pdf()`` then uses this new value to calculate the
pdf.

.. jupyter-execute::

    # Stage 3: minimise the given negative likelihood but floating only specific parameters (e.g. mu)
    result2 = minimizer.minimize(nll, params=[mu])

It is important to highlight that conceptually zfit separates the minimisation of the loss function with respect to the error calculation,
in order to give the freedom of calculating this error whenever needed and to allow the use of external error calculation packages.

In order to get an estimate for the errors, it is possible to call ``Hesse`` that will calculate
the parameter uncertainties. This uses the inverse Hessian to approximate the minimum of the loss and returns a symmetric estimate.
When using weighted datasets, this will automatically perform the asymptotic correction to the fit covariance matrix,
returning corrected parameter uncertainties to the user. The correction applied is based on Equation 18 in `this paper <https://arxiv.org/abs/1911.01303>`_.

To call ``Hesse``, do:

.. jupyter-execute::

    param_hesse = result.hesse()
    print(param_hesse)

which will return a dictionary of the fit parameters as keys with ``error`` values for each one.
The errors will also be added to the result object and show up when printing the result.

While the hessian approximation has many advantages, it may not hold well for certain loss functions, especially for
asymetric uncertainties. It is also possible to use a more CPU-intensive error calculating with the ``errors`` method.
This has the advantage of taking into account all the correlations and can describe well a
a loss minimum that is not well approximated by a quadratic function *(it is however not valid in the case of weights and takes
considerably longer).* It estimates the lower and upper uncertainty independently.
As an example, with the :py:class:`~zfit.minimize.Minuit` one can calculate the ``MINOS`` uncertainties with:

.. jupyter-execute::
    :hide-output:

    param_errors, _ = result.errors()

.. jupyter-execute::

    print(param_errors)


Once we've performed the fit and obtained the corresponding uncertainties,
it is now important to examine the fit results.
The object ``result`` (:py:class:`~zfit.minimizers.fitresult.FitResult`) has all the relevant information we need:

.. jupyter-execute::

    print(f"Function minimum: {result.fmin}")
    print(f"Converged: {result.converged}")
    print(f"Valid: {result.valid}")

This is all available if we print the fitresult (not shown here as display problems).

.. jupyter-execute::
    :hide-output:

    print(result)

Similarly one can obtain only the information on the fitted parameters with

.. jupyter-execute::

    # Information on all the parameters in the fit
    print(result.params)

    # Printing information on specific parameters, e.g. mu
    print("mu={}".format(result.params[mu]['value']))


As already mentioned, there is no dedicated plotting feature within zfit. However, we can easily use external
libraries, such as ``matplotlib`` and `mplhep, a library for HEP-like plots <https://github.com/scikit-hep/mplhep>`_ ,
to do the job:

.. jupyter-execute::

    import mplhep
    import matplotlib.pyplot as plt
    import numpy as np

    lower, upper = obs.limits
    data_np = zfit.run(data.value()[:, 0])

    # plot the data as a histogramm
    bins = 80
    counts, bin_edges = np.histogram(data_np, bins, range=(lower[-1][0], upper[0][0]))
    mplhep.histplot((counts, bin_edges), yerr=True, color='black', histtype='errorbar')

    # evaluate the func at multiple x and plot
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    y_plot = zfit.run(gauss.pdf(x_plot, norm_range=obs))
    plt.plot(x_plot, y_plot * data_np.shape[0] / bins * obs.area(), color='xkcd:blue')
    plt.show()


The specific call to :func:`zfit.run` simply converts the Eager Tensor (that is already array-like) to a Numpy array.
Often, this conversion is however not necessary and a Tensor can directly be used.

The full script :jupyter-download:script:`can be downloaded here <5 minutes to zfit>`.
