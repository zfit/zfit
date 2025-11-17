Loss
----

The loss, or also called "cost", describes the disagreement between the data and the model.
Most commonly, the likelihood (or, to be precise, the *negative log* likelihood)
is used, as the maximum likelihood estimation provides many
beneficial characteristics.

Binned losses require the PDF and data to be binned as well.

Extended losses take the expected count ("yield") of a PDF into account and require the
PDF to be extended in the first place.

.. autosummary::
    :toctree: _generated/loss

    zfit.loss.UnbinnedNLL
    zfit.loss.ExtendedUnbinnedNLL
    zfit.loss.BinnedNLL
    zfit.loss.ExtendedBinnedNLL
    zfit.loss.BinnedChi2
    zfit.loss.ExtendedBinnedChi2
    zfit.loss.BaseLoss
    zfit.loss.SimpleLoss
