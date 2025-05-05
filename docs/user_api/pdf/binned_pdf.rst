Binned PDFs
###########

Binned PDFs extend the functionality of unbinned PDFs by providing more histogram-like features in
addition to the basic unbinned PDFs. They interface with the
`Unified Histogram Interface (uhi) <https://uhi.readthedocs.io/en/latest/?badge=latest>`_
that is provided `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/>`_ and especially
`Hist <https://github.com/scikit-hep/hist>`_.

Below are the available binned PDFs:

Histogram PDF
-----------

The :py:class:`~zfit.pdf.HistogramPDF` creates a PDF from a histogram, preserving the bin structure and values.

Binwise Scale Modifier
------------------

The :py:class:`~zfit.pdf.BinwiseScaleModifier` allows modifying individual bins of a binned PDF with scale factors.

Binned From Unbinned PDF
--------------------

The :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF` converts an unbinned PDF to a binned representation.

Spline Morphing PDF
---------------

The :py:class:`~zfit.pdf.SplineMorphingPDF` creates a morphing between different template histograms using spline interpolation.

Binned Sum PDF
----------

The :py:class:`~zfit.pdf.BinnedSumPDF` combines multiple binned PDFs with different fractions.

Spline PDF
-------

The :py:class:`~zfit.pdf.SplinePDF` creates a PDF from spline interpolation between points.

Unbinned From Binned PDF
-------------------

The :py:class:`~zfit.pdf.UnbinnedFromBinnedPDF` converts a binned PDF to an unbinned representation.

.. autosummary::
    :toctree: _generated/binned_pdf

    zfit.pdf.HistogramPDF
    zfit.pdf.BinwiseScaleModifier
    zfit.pdf.BinnedFromUnbinnedPDF
    zfit.pdf.SplineMorphingPDF
    zfit.pdf.BinnedSumPDF
    zfit.pdf.SplinePDF
    zfit.pdf.UnbinnedFromBinnedPDF
