Loss
----

The loss, or also called "cost", describes the disagreement between the data and the model.
Most commonly, the likelihood (or, to be precise, the *negative log* likelihood)
is used, as the maximum likelihood estimation provides many
beneficial characteristics.

Binned losses require the PDF and data to be binned as well.

Extended losses take the expected count ("yield") of a PDF into account and require the
PDF to be extended in the first place.

**Available Loss Classes:**

* :py:class:`zfit.loss.UnbinnedNLL`
* :py:class:`zfit.loss.ExtendedUnbinnedNLL`
* :py:class:`zfit.loss.BinnedNLL`
* :py:class:`zfit.loss.ExtendedBinnedNLL`
* :py:class:`zfit.loss.BinnedChi2`
* :py:class:`zfit.loss.ExtendedBinnedChi2`
* :py:class:`zfit.loss.BaseLoss`
* :py:class:`zfit.loss.SimpleLoss`
