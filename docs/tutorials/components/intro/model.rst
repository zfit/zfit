.. _basic-model:


Building a model
================

In order to build a generic model the concept of function and distributed density functions (PDFs) need to be clarified.
The PDF, or density of a continuous random variable, of X is a function f(x) that describes the relative likelihood for this random variable to take on a given value.
In this sense, for any two numbers a and b with :math:`a \leq b`,

:math:`P(a \leq X \leq b) = \int^{b}_{a}f(X)dx`

That is, the probability that X takes on a value in the interval :math:`[a, b]` is the area above this interval and under the graph of the density function.
In other words, in order to a function to be a PDF it must satisfy two criteria:
1. :math:`f(x) \geq 0` for all x;
2. :math:`\int^{\infty}_{-\infty}f(x)dx =` are under the entire graph of :math:`f(x)=1`.
In zfit these distinctions are respected, *i.e.*, a function can be converted into a PDF by imposing the basic two criteria above.

Predefined PDFs and basic properties
------------------------------------

A series of predefined PDFs are available to the users and can be easily accessed using autocompletion (if available). In fact, all of these can also be seen in

.. jupyter-kernel::
  :id: zfit_model_introduction.ipynb

.. jupyter-execute::
    :hide-output:
    :hide-code:

    import os
    os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"

    import zfit
    from zfit import z
    import numpy as np


.. thebe-button:: Run interactively

.. jupyter-execute::

    print(zfit.pdf.__all__)

These include the basic function but also some operations discussed below. Let's consider
the simple example of a ``CrystalBall``.
PDF objects must also be initialised giving their named parameters. For example:

.. jupyter-execute::

    obs = zfit.Space('x', limits=(4800, 6000))

    # Creating the parameters for the crystal ball
    mu = zfit.Parameter("mu", 5279, 5100, 5300)
    sigma = zfit.Parameter("sigma", 20, 0, 50)
    a = zfit.Parameter("a", 1, 0, 10)
    n = zfit.Parameter("n", 1, 0, 10)

    # Single crystal Ball
    model_cb = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a, n=n)

In this case the CB object corresponds to a normalised PDF. The main properties of a PDF, e.g.
the probability for a given normalisation range or even
to set a temporary normalisation range can be given as

.. jupyter-execute::

    # Get the probabilities of some random generated events
    probs = model_cb.pdf(x=np.random.random(10))
    # And now execute the tensorflow graph
    result = zfit.run(probs)
    print(result)

.. jupyter-execute::

    # The norm range of the pdf can be changed any time with a contextmanager (temporary) or without (permanent)
    with model_cb.set_norm_range((5000, 6000)):
        print(model_cb.norm_range)

Another feature for the PDF is to calculate its integral in a certain limit. This can be easily achieved by

.. jupyter-execute::

    # Calculate the integral between 5000 and 5250 over the PDF normalized
    integral_norm = model_cb.integrate(limits=(5000, 5250))
    print(f"Integral={integral_norm}")

In this case the CB has been normalised using the range defined in the observable.
Conversely, the ``norm_range`` in which the PDF is normalised can also be specified as input.

Composite PDF
-------------

A common feature in building composite models it the ability to combine in terms of sum and products different PDFs.
There are two ways to create such models, either with the class API or with simple Python syntax.
Let's consider a second crystal ball with the same mean position and width, but different tail parameters

.. jupyter-execute::

    # New tail parameters for the second CB
    a2 = zfit.Parameter("a2", -1, -10, 0)
    n2 = zfit.Parameter("n2", 1, 0, 10)

    # New crystal Ball function defined in the same observable range
    model_cb2 = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a2, n=n2)

We can now combine these two PDFs to create a double Crystal Ball with a single mean and width through the :py:class:`zfit.pdf.SumPDF` class:

.. jupyter-execute::

    # or via the class API
    frac = 0.3  # can also be a Parameter
    double_cb_class = zfit.pdf.SumPDF(pdfs=[model_cb, model_cb2], fracs=frac)

Notice that the new PDF has the same observables as the original ones, as they coincide.
Alternatively one could consider having PDFs for different axis, which would then create a totalPDF with higher dimension.

A simple extension of these operations is if we want to instead of a sum of PDFs, to model a two-dimensional Gaussian (e.g.):

.. jupyter-execute::

    # Defining two Gaussians in two different observables
    mu1 = zfit.Parameter("mu1", 1.)
    sigma1 = zfit.Parameter("sigma1", 1.)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)

    obs2 = zfit.Space('y', limits=(5, 11))

    mu2 = zfit.Parameter("mu2", 1.)
    sigma2 = zfit.Parameter("sigma2", 1.)
    gauss2 = zfit.pdf.Gauss(obs=obs2, mu=mu2, sigma=sigma2)

    # Producing the product of two PDFs
    prod_gauss = gauss1 * gauss2
    # Or alternatively
    prod_gauss_class = zfit.pdf.ProductPDF(pdfs=[gauss2, gauss1])  # notice the different order or the pdf

The new PDF is now in two dimensions.
The order of the observables follows the order of the PDFs given.

.. jupyter-execute::

    print("python syntax product obs", prod_gauss.obs)

.. jupyter-execute::

    print("class API product obs", prod_gauss_class.obs)


Extended PDF
------------

In the event there are different *species* of distributions in a given observable,
the simple sum of PDFs does not a priori provides the absolute number of events for each specie but rather the fraction as seen above.
An example is a Gaussian mass distribution with an exponential background, e.g.

:math:`P = f_{S}\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} + (1 - f_{S}) e^{-\alpha x}`

Since we are interested to express a measurement of the number of events,
the expression :math:`M(x) = N_{S}S(x) + N_{B}B(x)` respect that M(x) is normalised to :math:`N_{S} + N_{B} = N` instead of one.
This means that :math:`M(x)` is not a true PDF but rather an expression for two quantities, the shape and the number of events in the distributions.

An extended PDF can be easily implemented in zfit in two ways:

.. jupyter-execute::

    # Create a parameter for the number of events
    yield_gauss = zfit.Parameter("yield_gauss", 100, 0, 1000)

    # Extended PDF using a predefined method
    extended_gauss = gauss1.create_extended(yield_gauss)

This will leave ``gauss1`` unextended while the ``extended_gauss`` is now extended. However, there are cases where
:meth:`~zfit.pdf.BasePDF.create_extended` may fails, such as if it can't copy the original PDF. A PDF can also be
extended in-place

.. jupyter-execute::

    print(f"Gauss is extended: {gauss1.is_extended}")
    gauss1.set_yield(yield_gauss)
    print(f"Gauss is extended: {gauss1.is_extended}")

.. note::

    An extended PDF in zfit *does not fundamentally alter the behavior*. Most importantly,
    **anything that works for a non-extended PDF will work in the exact same way if the PDF is extended** (anything
    working, e.g. exceptions may differ). This implies that the output of :meth:`~zfit.pdf.BasePDF.pdf` and
    :meth:`~zfit.pdf.BasePDF.integrate` will remain the same.

    An extended PDF will have *more* available functionality such as the methods :meth:`~zfit.pdf.BasePDF.ext_pdf` and
    :meth:`~zfit.pdf.BasePDF.ext_integrate`, which will scale the output by the yield.

    This means that there is no damage done in extending a PDF. It also implies that the other way around,
    "de-extending" is not possible but also never required.

Custom PDF
----------
A fundamental design choice of zfit is the ability to create custom PDFs and functions in an easy way.
Let's consider a simplified implementation


.. jupyter-execute::

    class MyGauss(zfit.pdf.ZPDF):
        """Simple implementation of a Gaussian similar to zfit.pdf.Gauss class"""
        _N_OBS = 1  # dimension, can be omitted
        _PARAMS = ['mean', 'std']  # name of the parameters

        def _unnormalized_pdf(self, x):
           x = z.unstack_x(x)
           mean = self.params['mean']
           std  = self.params['std']
           return z.exp(- ((x - mean) / std) ** 2)

This is the basic information required for this custom PDF.
With this new PDF one can access the same feature of the predefined PDFs, e.g.

.. jupyter-execute::

    obs_own = zfit.Space("my obs", limits=(-4, 4))

    mean = zfit.Parameter("mean", 1.)
    std  = zfit.Parameter("std", 1.)
    my_gauss = MyGauss(obs=obs_own, mean=mean, std=std)


    # For instance sampling, integral and probabilities
    data     = my_gauss.sample(15)
    integral = my_gauss.integrate(limits=(-1, 2))
    probs    = my_gauss.pdf(data,norm_range=(-3, 4))
    print(f"Probs: {probs} and integral: {integral}")

Finally, we could also improve the description of the PDF by providing a analytical integral for the ``MyGauss`` PDF:

.. jupyter-execute::

    def gauss_integral_from_any_to_any(limits, params, model):
       (lower,), (upper,) = limits.limits
       mean = params['mean']
       std = params['std']
       # Write you integral
       return 42. # Dummy value

    # Register the integral
    limits = zfit.Space(axes=0, limits=(zfit.Space.ANY_LOWER, zfit.Space.ANY_UPPER))
    MyGauss.register_analytic_integral(func=gauss_integral_from_any_to_any, limits=limits)


Sampling from a Model
'''''''''''''''''''''

In order to sample from model, there are two different methods,
:py:meth:`~zfit.core.basemodel.BaseModel.sample` for **advanced** sampling returning a Tensor, and
:py:meth:`~zfit.core.basemodel.BaseModel.create_sampler` for **multiple sampling** as used for toys.

Tensor sampling
'''''''''''''''''

The sample from :py:meth:`~zfit.core.basemodel.BaseModel.sample` is a Tensor that samples when executed.
This is for an advanced usecase only

Advanced sampling and toy studies
'''''''''''''''''''''''''''''''''''''

More advanced and repeated sampling, such as used in toy studies, will be
explained in :ref:`playing_with_toys`.



Download this tutorial :jupyter-download:notebook:`notebook <zfit_model_introduction.ipynb>`,
:jupyter-download:script:`script <zfit_model_introduction.ipynb>`
