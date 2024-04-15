.. _playing_with_toys:

Toy studies and inference
================================================

While a single fit is useful, it does not say a lot about the *uncertainty* of
the result and whether the fit is biased in any way or not.

Many statistical methods, such as obtaining sWeights,
(Feldman and Cousins) Confidence Interval, setting limits and more are
all covered it the
`hepstats library <https://github.com/scikit-hep/hepstats>`_,
which work directly with zfit parts.

For other toy studies, models offer a sampler function that can be used
for repeated sampling.

Playing with toys: Multiple samplings
''''''''''''''''''''''''''''''''''''''

The method :py:meth:`~zfit.core.basemodel.BaseModel.create_sampler` returns a sampler that can be used
like a :py:class:`~zift.Data` object (e.g. for building a :py:class:`~zfit.core.interfaces.ZfitLoss`).
The sample generated depends on the original pdf at this point, e.g. parameters have the
value they have when :py:meth:`~zfit.core.basemodel.BaseModel.create_sampler` is called, respectively the values that are given explicitly as ``params`` to it.

Reusing the model, obs and parameters from :ref:`basic-model`,
this is typically how toys look like:


.. jupyter-execute::
    :hide-output:
    :hide-code:

    import os
    os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"

    import zfit
    from zfit import z
    import numpy as np

    obs = zfit.Space('x', -10, 10)

    mu1 = zfit.Parameter("mu1", 1)
    sigma1 = zfit.Parameter("sigma1", 1.3)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)

    mu2 = zfit.Parameter("mu2", 3.)
    sigma2 = zfit.Parameter("sigma2", 2.)

.. jupyter-execute::

    # using the previous gaussians and obs to create a model

    gauss3 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
    model = zfit.pdf.SumPDF([gauss1, gauss3], fracs=0.4)

    sampler = model.create_sampler(n=1000)
    nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)

    minimizer = zfit.minimize.Minuit()

    results = []
    nruns = 5
    for run_number in range(nruns):
        sampler.resample()  # now the resampling gets executed


        # initialize the parameters randomly
        mu1.set_value(np.random.normal())
        sigma1.set_value(abs(np.random.normal()) + 0.5)

        result = minimizer.minimize(nll)
        results.append(result)

        # safe the result, collect the values, calculate errors...





If some parameters are constrained to values observed from external measurements, usually Gaussian constraints,
then sampling of the observed values might be needed to obtain an unbiased sample from the model. Example:

TODO: the sample below is not correct and needs updating...
