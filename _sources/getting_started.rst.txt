=============================
Getting started with zfit
=============================

The ``zfit`` library provides a simple model fitting and sampling framework for a broad list of applications. This section is designed to give an overview of the main concepts and features in the context of likelihood fits in a *crash course* manner. The simplest example is to generate, fit and plot a Gaussian distribution.  

The first step is to naturally import ``zfit`` and verify if the installation has been done successfully (plus some additional imports of helpful libraries): 

.. code-block:: python

    import tensorflow as tf
    import zfit
    from zfit import ztf
    print("TensorFlow version:", tf.__version__)

Since we want to generate/fit a Gaussian within a given range, an observable space defines the domain of the PDF. This can be created using the :py:class:`~zfit.Space` class

.. code-block:: python

    obs = zfit.Space('x', limits=(-10, 10))

The best interpretation of the observable at this stage is that it defines the name axis. 

Using this domain, we can now create a simple Gaussian PDF. There are already pre-defined the most common PDFs within the :py:class`~zfit.pdf` class, including a simple Gaussian. First, we have to define the parameters of the PDF and their limits

.. code-block:: python

    mu    = zfit.Parameter("mu"   , 2.4, -1.0, 5.)
    sigma = zfit.Parameter("sigma", 1.3,  0.0, 5.)

With these parameters we can instantiate the Gaussian PDF from the library

.. code-block:: python

    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

It is important to follow the pre-defined structure of the given PDF. In this case, ``obs``, ``mu`` and ``sigma`` have to be passed as arguments to the ``Gauss`` PDF. 

The next stage is to create a dataset to be fitted. There are several ways of producing this within the ``zfit`` framework, however, for simplicity we simply produce it using numpy, e.g.

.. code-block:: python

    mu_true = 0.
    sigma_true = 1.
    data_np = np.random.normal(mu_true, sigma_true, size=10000)
    data = zfit.data.Data.from_numpy(obs=obs, array=data_np)
    print(data)

Now we have all the ingredients in order to perform a maximum likelihood fit. Conceptually this corresponds to three basic steps: (1) create a negative likelihood (i.e. $\log\mathcal{L}$); (2) instantiate a given minimiser; (3) and minimise the likelihood. 

.. code-block:: python
    # Stage 1: create an unbinned likelihood with the given PDF and dataset 
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit
    minimizer = zfit.minimize.MinuitMinimizer()

    # Stage 3: minimise the given negative likelihood
    result = minimizer.minimize(nll)

This corresponds to the most basic example where the negative likelihood is defined within the pre-determined observable range and all the parameters in the PDF are floated in the fit. 



# Get the fitted values, again by run the variable graphs
params = minimum.params

print(params)

%matplotlib inline  

import matplotlib.pyplot as plt
n_bins = 50
range_ = (-5,5)
_ = plt.hist(data_np, bins=n_bins, range=range_)
x = np.linspace(*range_, num=1000)
pdf = zfit.run(gauss.pdf(x, norm_range=(-10, 10)))
#_ = plt.plot(x, data_np.shape[0]/n_bins*10.*pdf[0,:])


The high level interface of zfit within TensorFlow
=============================

This object object contains all the functions you would expect from a PDF, such as calculating a probability, calculating its integral, etc. An example is if we want to get the probability for This can be visualised by 

# Now, get some probability values
# The probs object is not executed yet
consts = [-1, 0, 1]
probs = gauss.pdf(ztf.constant(consts), norm_range=(-np.infty, np.infty))
# And now execute the tensorflow graph
result = zfit.run(probs)
print("x values: {}\nresult:   {}".format(consts, result))

**NB**: Currently, one important caveat is that all `zfit` objects are based on `tensorflow`, and therefore they are graphs that are not executed immediately, but need to be run on a session:

```python
zfit.run(TensorFlow_object)
```

Here, we can see the power of the context managers used to change the normalisation range.

with gauss.set_norm_range((-1e6, 1e6)):  # play around with different norm ranges
# with gauss.set_norm_range((-100, 100)):
    print(zfit.run(gauss.integrate((-0.6, 0.6))))
    print(zfit.run(gauss.integrate((-3, 3))))
    print(zfit.run(gauss.integrate((-100, 100))))


