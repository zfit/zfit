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
The sampling itself is *not yet done* but only when :py:meth:`~zfit.core.data.Sampler.resample` is
invoked. The sample generated depends on the original pdf at this point, e.g. parameters have the
value they have when the :py:meth:`~zfit.core.data.Sampler.resample` is invoked. To have certain
parameters fixed, they have to be specified *either* on :py:meth:`~zfit.core.basemodel.BaseModel.create_sampler`
via ``fixed_params``, on :py:meth:`~zfit.core.data.Sampler.resample` by specifying which parameter
will take which value via ``param_values`` or by changing the attribute of :py:class:`~zfit.core.data.Sampler`.

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

    obs = zfit.Space('x', limits=(4800, 6000))

    mu1 = zfit.Parameter("mu1", 1.)
    sigma1 = zfit.Parameter("sigma1", 1.)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)

    mu2 = zfit.Parameter("mu2", 1.)
    sigma2 = zfit.Parameter("sigma2", 1.)

.. jupyter-execute::

    # using the previous gaussians and obs to create a model

    gauss3 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
    model = zfit.pdf.SumPDF([gauss1, gauss3], fracs=0.4)

    sampler = model.create_sampler(n=1000,fixed_params=True)
    nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)

    minimizer = zfit.minimize.Minuit()

    results = []
    nruns = 5
    for run_number in range(nruns):
        # initialize the parameters randomly
        sampler.resample()  # now the resampling gets executed

        mu1.set_value(np.random.normal())
        sigma1.set_value(abs(np.random.normal()) + 0.5)

        result = minimizer.minimize(nll)
        results.append(result)

        # safe the result, collect the values, calculate errors...

Here we fixed all parameters as they have been initialized and then sample. If we do not provide any
arguments to ``resample``, this will always sample now from the distribution with the parameters set to the values when
the sampler was created.


To give another, though not very useful example:

.. jupyter-execute::

    # create a model depending on mu1, sigma1, mu2, sigma2

    sampler = model.create_sampler(n=1000, fixed_params=[mu1, mu2])
    nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)

    sampler.resample()  # now it sampled

    # do something with nll
    minimizer.minimize(nll)  # minimize

    sampler.resample()
    # note that the nll, being dependent on ``sampler``, also changed!

The sample is now resampled with the *current values* (minimized values) of ``sigma1``, ``sigma2`` and with
the initial values of ``mu1``, ``mu2`` (because they have been fixed).

We can also specify the parameter values explicitly by
using the following argument. Reusing the example above

.. jupyter-execute::

    sigma1.set_value(np.random.normal())
    sampler.resample(param_values={sigma1: 5})

The sample (and therefore also the sample the ``nll`` depends on) is now sampled with ``sigma1`` set to 5.

If some parameters are constrained to values observed from external measurements, usually Gaussian constraints,
then sampling of the observed values might be needed to obtain an unbiased sample from the model. Example:

.. jupyter-execute::

    # same model depending on mu1, sigma1, mu2, sigma2

    constraint = zfit.constraint.GaussianConstraint(params=[sigma1, sigma2],
                                                    observation=[1.0, 0.5],
                                                    uncertainty=[0.1, 0.05])

    n_samples = 5

    sampler = model.create_sampler(n=n_samples, fixed_params=[mu1, mu2])
    nll = zfit.loss.UnbinnedNLL(model=model, data=sampler, constraints=constraint)

    constr_values = constraint.sample(n=n_samples)

    for constr_params, constr_vals in constr_values.items():
        sampler.resample()
        # do something with nll, temporarily assigning values to the parameters
        with zfit.param.set_values(constr_params, constr_vals):
            minimizer.minimize(nll)  # minimize
