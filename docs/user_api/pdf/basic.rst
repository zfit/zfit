Basic shapes are fundamendal PDFs, with often well-known functional form.
They are usually fully analytically implemented and often a thin
wrapper around :py:class:`~tensorflow_probability.distribution.Distribution`.
Any missing shape can be easily wrapped using :py:class:`~zfit.pdf.WrapDistribution`.

.. autosummary::
    :toctree: _generated/basic

    zfit.pdf.Gauss
    zfit.pdf.Exponential
    zfit.pdf.CrystalBall
    zfit.pdf.DoubleCB
    zfit.pdf.GeneralizedCB
    zfit.pdf.GaussExpTail
    zfit.pdf.GeneralizedGaussExpTail
    zfit.pdf.Uniform
    zfit.pdf.Cauchy
    zfit.pdf.Voigt
    zfit.pdf.TruncatedGauss
    zfit.pdf.BifurGauss
    zfit.pdf.Poisson
    zfit.pdf.LogNormal
    zfit.pdf.QGauss
    zfit.pdf.ChiSquared
    zfit.pdf.StudentT
    zfit.pdf.Gamma
    zfit.pdf.JohnsonSU
