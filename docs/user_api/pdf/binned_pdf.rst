Binned PDFs
###########

Binned PDFs extend the functionality of unbinned PDFs by providing more histogram-like features in
addition to the basic unbinned PDFs. They interface with the
`Unified Histogram Interface (uhi) <https://uhi.readthedocs.io/en/latest/?badge=latest>`_
that is provided `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/>`_ and especially
`Hist <https://github.com/scikit-hep/hist>`_.


.. autosummary::
    :toctree: _generated/binned_pdf

    zfit.pdf.HistogramPDF
    zfit.pdf.BinwiseScaleModifier
    zfit.pdf.BinnedFromUnbinnedPDF
    zfit.pdf.SplineMorphingPDF
    zfit.pdf.BinnedSumPDF
    zfit.pdf.SplinePDF
    zfit.pdf.UnbinnedFromBinnedPDF
