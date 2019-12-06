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
In zfit these distinctions are respected, *i.e.*, a function can be converted into a PDF by imposing the basic two criteria above... _basic-model:

Predefined PDFs and basic properties
------------------------------------

A series of predefined PDFs are available to the users and can be easily accessed using autocompletion (if available). In fact, all of these can also be seen in

.. code-block:: pycon

    >>> print(zfit.pdf.__all__)
    ['BasePDF', 'Exponential', 'CrystalBall', 'Gauss', 'Uniform', 'WrapDistribution', 'ProductPDF', 'SumPDF', 'ZPDF', 'SimplePDF', 'SimpleFunctorPDF']

These include the basic function but also some operations discussed below. Let's consider the simple example of a ``CrystalBall``.
PDF objects must also be initialised giving their named parameters. For example:

.. code-block:: pycon

    >>> obs = zfit.Space('x', limits=(4800, 6000))

    >>> # Creating the parameters for the crystal ball
    >>> mu = zfit.Parameter("mu", 5279, 5100, 5300)
    >>> sigma = zfit.Parameter("sigma", 20, 0, 50)
    >>> a = zfit.Parameter("a", 1, 0, 10)
    >>> n = zfit.Parameter("n", 1, 0, 10)

    >>> # Single crystal Ball
    >>> model_cb = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a, n=n)

In this case the CB object corresponds to a normalised PDF. The main properties of a PDF, e.g. the probability for a given normalisation range or even
to set a temporary normalisation range can be given as

.. code-block:: pycon

    >>> # Get the probabilities of some random generated events
    >>> probs = model_cb.pdf(x=np.random.random(10), norm_range=(5100, 5400))
    >>> # And now execute the tensorflow graph
    >>> result = zfit.run(probs)
    >>> print(result)
    [3.34187765e-05 3.34196917e-05 3.34202989e-05 3.34181458e-05
     3.34172973e-05 3.34209238e-05 3.34164538e-05 3.34210950e-05
     3.34201199e-05 3.34209360e-05]

    >>> # The norm range of the pdf can be changed any time by
    >>> model_cb.set_norm_range((5000, 6000))

Another feature for the PDF is to calculate its integral in a certain limit. This can be easily achieved by

.. code-block:: pycon

    >>> # Calculate the integral between 5000 and 5250 over the PDF normalized
    >>> integral_norm = model_cb.integrate(limits=(5000, 5250))

In this case the CB has been normalised using the range defined in the observable.
Conversely, the ``norm_range`` in which the PDF is normalised can also be specified as input.

Composite PDF
-------------

A common feature in building composite models it the ability to combine in terms of sum and products different PDFs.
There are two ways to create such models, either with the class API or with simple Python syntax.
Let's consider a second crystal ball with the same mean position and width, but different tail parameters

.. code-block:: pycon

    >>> # New tail parameters for the second CB
    >>> a2 = zfit.Parameter("a2", -1, 0, -10)
    >>> n2 = zfit.Parameter("n2", 1, 0, 10)

    >>> # New crystal Ball function defined in the same observable range
    >>> model_cb2 = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a2, n=n2)

We can now combine these two PDFs to create a double Crystal Ball with a single mean and width, either using arithmetic operations

.. code-block:: pycon

    >>> # First needs to define a parameters that represent
    >>> # the relative fraction between the two PDFs
    >>> frac = zfit.Parameter("frac", 0.5, 0, 1)

    >>> # Two different ways to combine
    >>> double_cb = frac * model_cb + model_cb2

Or through the :py:class:`zfit.pdf.SumPDF` class:

.. code-block:: pycon

    >>> # or via the class API
    >>> double_cb_class = zfit.pdf.SumPDF(pdfs=[model_cb, model_cb2], fracs=frac)

Notice that the new PDF has the same observables as the original ones, as they coincide.
Alternatively one could consider having PDFs for different axis, which would then create a totalPDF with higher dimension.

A simple extension of these operations is if we want to instead of a sum of PDFs, to model a two-dimensional Gaussian (e.g.):

.. code-block:: pycon

    >>> # Defining two Gaussians in two different axis (obs)
    >>> mu1 = zfit.Parameter("mu1", 1.)
    >>> sigma1 = zfit.Parameter("sigma1", 1.)
    >>> gauss1 = zfit.pdf.Gauss(obs="obs1", mu=mu1, sigma=sigma1)

    >>> mu2 = zfit.Parameter("mu2", 1.)
    >>> sigma2 = zfit.Parameter("sigma2", 1.)
    >>> gauss2 = zfit.pdf.Gauss(obs="obs2", mu=mu2, sigma=sigma2)

    >>> # Producing the product of two PDFs
    >>> prod_gauss = gauss1 * gauss2
    >>> # Or alternatively
    >>> prod_gauss_class = zfit.pdf.ProductPDF(pdfs=[gauss2, gauss1])  # notice the different order or the pdf

The new PDF is now in two dimensions.
The order of the observables follows the order of the PDFs given.

.. code-block:: pycon

    >>> print("python syntax product obs", prod_gauss.obs)
    [python syntax product obs ('obs1', 'obs2')]
    >>> print("class API product obs", prod_gauss_class.obs)
    [class API product obs ('obs2', 'obs1')]


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

.. code-block:: pycon

    >>> # Create a parameter for the number of events
    >>> yieldGauss = zfit.Parameter("yieldGauss", 100, 0, 1000)

    >>> # Extended PDF using a predefined method
    >>> extended_gauss_method = gauss.create_extended(yieldGauss)
    >>> # Or simply with a Python syntax of multiplying a PDF with the parameter
    >>> extended_gauss_python = yieldGauss * gauss


Custom PDF
----------
A fundamental design choice of zfit is the ability to create custom PDFs and functions in an easy way.
Let's consider a simplified implementation


.. code-block:: pycon

    >>> class MyGauss(zfit.pdf.ZPDF):
    ...    """Simple implementation of a Gaussian similar to :py:class`~zfit.pdf.Gauss` class"""
    ...    _N_OBS = 1  # dimension, can be omitted
    ...    _PARAMS = ['mean', 'std']  # the name of the parameters

    >>> def _unnormalized_pdf(self, x):
    ...    x = zfit.ztf.unstack_x(x)
    ...    mean = self.params['mean']
    ...    std  = self.params['std']
    ...    return zfit.ztf.exp(- ((x - mean)/std)**2)

This is the basic information required for this custom PDF.
With this new PDF one can access the same feature of the predefined PDFs, e.g.

.. code-block:: pycon

    >>> obs = zfit.Space("obs1", limits=(-4, 4))

    >>> mean = zfit.Parameter("mean", 1.)
    >>> std  = zfit.Parameter("std", 1.)
    >>> my_gauss = MyGauss(obs='obs1', mean=mean, std=std)

    >>> # For instance integral probabilities
    >>> integral = my_gauss.integrate(limits=(-1, 2))
    >>> probs    = my_gauss.pdf(data, norm_range=(-3, 4))

Finally, we could also improve the description of the PDF by providing a analytical integral for the ``MyGauss`` PDF:

.. code-block:: pycon

    >>> def gauss_integral_from_any_to_any(limits, params, model):
    ...    (lower,), (upper,) = limits.limits
    ...    mean = params['mean']
    ...    std = params['std']
    ...    # Write you integral
    ...    return 42. # Dummy value

    >>> # Register the integral
    >>> limits = zfit.Space.from_axes(axes=0, limits=(zfit.Space.ANY_LOWER, zfit.Space.ANY_UPPER))
    >>> MyGauss.register_analytic_integral(func=gauss_integral_from_any_to_any, limits=limits)


Sampling from a Model
'''''''''''''''''''''

In order to sample from model, there are two different methods,
:py:meth:`~zfit.core.basemodel.BaseModel.sample` for **advanced** sampling returning a Tensor, and
:py:meth:`~zfit.core.basemodel.BaseModel.create_sampler` for **multiple sampling** as used for toys.

Tensor sampling
'''''''''''''''''

The sample from :py:meth:`~zfit.core.basemodel.BaseModel.sample` is a Tensor that samples when executed.
This is for an advanced usecase only

Playing with toys: Multiple samplings
'''''''''''''''''''''''''''''''''''''

The method :py:meth:`~zfit.core.basemodel.BaseModel.create_sampler` returns a sampler that can be used
like a :py:class:`~zift.Data` object (e.g. for building a :py:class:`~zfit.core.interfaces.ZfitLoss`).
The sampling itself is *not yet done* but only when :py:meth:`~zfit.core.data.Sampler.resample` is
invoked. The sample generated depends on the original pdf at this point, e.g. parameters have the
value they have when the :py:meth:`~zfit.core.data.Sampler.resample` is invoked. To have certain
parameters fixed, they have to be specified *either* on :py:meth:`~zfit.core.basemodel.BaseModel.create_sampler`
via `fixed_params`, on :py:meth:`~zfit.core.data.Sampler.resample` by specifying which parameter
will take which value via `param_values` or by changing the attribute of :py:class:`~zfit.core.data.Sampler`.

How typically toys look like:
.. _playing_with_toys:

A typical example of toys would therefore look like

.. code:: pycon

    >>> # create a model depending on mu, sigma

    >>> sampler = model.create_sampler(n=1000, fixed_params=True)
    >>> nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)

    >>> minimizer = zfit.minimize.Minuit()

    >>> for run_number in n_runs:
    ...    # initialize the parameters randomly
    ...    sampler.resample()  # now the resampling gets executed
    ...
    ...    mu.set_value(np.random.normal())
    ...    sigma.set_value(abs(np.random.normal()))
    ...
    ...    result = minimizer.minimize(nll)
    ...
    ...    # safe the result, collect the values, calculate errors...

Here we fixed all parameters as they have been initialized and then sample. If we do not provide any
arguments to `resample`, this will always sample now from the distribution with the parameters set to the
 values when the sampler was created.


To give another, though not very useful example:

.. code:: pycon

    >>> # create a model depending on mu1, sigma1, mu2, sigma2

    >>> sampler = model.create_sampler(n=1000, fixed_params=[mu1, mu2])
    >>> nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)

    >>> sampler.resample()  # now it sampled

    >>> # do something with nll
    >>> minimizer.minimize(nll)  # minimize

    >>> sampler.resample()
    >>> # note that the nll, being dependent on `sampler`, also changed!

The sample is now resampled with the *current values* (minimized values) of `sigma1`, `sigma2` and with
the initial values of `mu1`, `mu2` (because they have been fixed).

We can also specify the parameter values explicitly by
using the following argument. Reusing the example above

.. code:: pycon

    >>> sigma.set_value(np.random.normal())
    >>> sampler.resample(param_values={sigma1: 5})

The sample (and therefore also the sample the `nll` depends on) is now sampled with `sigma1` set to 5.

If some parameters are constrained from external measurements, usually Gaussian constraints, then sampling of
those parameters might be needed to obtain an unbiased sample from the model. Example:

.. code:: pycon

    >>> # same model depending on mu1, sigma1, mu2, sigma2

    >>> constraint = zfit.constraint.GaussianConstraint(params=[sigma1, sigma2], mu=[1.0, 0.5], sigma=[0.1, 0.05])

    >>> n_samples = 1000

    >>> sampler = model.create_sampler(n=n_samples, fixed_params=[mu1, mu2])
    >>> nll = zfit.loss.UnbinnedNLL(model=model, data=sampler, constraints=constraint)

    >>> constr_values = constraint.sample(n=n_samples)

    >>> for i in range(n_samples):
    >>>     sampler.resample(param_values={sigma1: constr_values[sigma1][i],
    >>>                                    sigma2: constr_values[sigma2][i]})
    >>>     # do something with nll
    >>>     minimizer.minimize(nll)  # minimize
