Binned PDFs
###########

Binned PDFs extend the functionality of unbinned PDFs by providing more histogram-like features in
addition to the basic unbinned PDFs. They interface with the
`Unified Histogram Interface (uhi) <https://uhi.readthedocs.io/en/latest/?badge=latest>`_
that is provided `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/>`_ and especially
`Hist <https://github.com/scikit-hep/hist>`_.

Below are the available binned PDFs:

Histogram PDF
----------------------

The :py:class:`~zfit.pdf.HistogramPDF` creates a PDF from a histogram, preserving the bin structure and values.

.. image:: ../../images/_generated/pdfs/histogrampdf_shapes.png
   :width: 80%
   :align: center
   :alt: HistogramPDF with different shapes

.. autosummary::

    zfit.pdf.HistogramPDF

Binwise Scale Modifier
-----------------------------

The :py:class:`~zfit.pdf.BinwiseScaleModifier` allows modifying individual bins of a binned PDF with scale factors.

.. image:: ../../images/_generated/pdfs/binwisescalemodifier_patterns.png
   :width: 80%
   :align: center
   :alt: BinwiseScaleModifier with different scale patterns

.. autosummary::

    zfit.pdf.BinwiseScaleModifier

Binned From Unbinned PDF
-------------------------------

The :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF` converts an unbinned PDF to a binned representation.

.. image:: ../../images/_generated/pdfs/binnedfromunbinnedpdf_comparison.png
   :width: 80%
   :align: center
   :alt: BinnedFromUnbinnedPDF comparison

.. autosummary::

    zfit.pdf.BinnedFromUnbinnedPDF

Spline Morphing PDF
--------------------------

The :py:class:`~zfit.pdf.SplineMorphingPDF` creates a morphing between different template histograms using spline interpolation.

.. image:: ../../images/_generated/pdfs/splinemorphingpdf_morphing.png
   :width: 80%
   :align: center
   :alt: SplineMorphingPDF with different parameter values

.. autosummary::

    zfit.pdf.SplineMorphingPDF

Binned Sum PDF
------------------------------

The :py:class:`~zfit.pdf.BinnedSumPDF` combines multiple binned PDFs with different fractions.

.. image:: ../../images/_generated/pdfs/binnedsumpdf_fractions.png
   :width: 80%
   :align: center
   :alt: BinnedSumPDF with different component fractions

.. autosummary::

    zfit.pdf.BinnedSumPDF

Spline PDF
-----------------

The :py:class:`~zfit.pdf.SplinePDF` creates a PDF from spline interpolation between points.

.. image:: ../../images/_generated/pdfs/splinepdf_shapes.pngIGNORE
   :width: 80%
   :align: center
   :alt: SplinePDF with different shapes

.. autosummary::

    zfit.pdf.SplinePDF

Unbinned From Binned PDF
------------------------------

The :py:class:`~zfit.pdf.UnbinnedFromBinnedPDF` converts a binned PDF to an unbinned representation.

.. image:: ../../images/_generated/pdfs/unbinnedfrombinnedpdf_comparison.png
   :width: 80%
   :align: center
   :alt: UnbinnedFromBinnedPDF comparison

.. autosummary::

    zfit.pdf.UnbinnedFromBinnedPDF

All Binned PDFs

.. autosummary::
    :toctree: _generated/binned_pdf

    zfit.pdf.HistogramPDF
    zfit.pdf.BinwiseScaleModifier
    zfit.pdf.BinnedFromUnbinnedPDF
    zfit.pdf.SplineMorphingPDF
    zfit.pdf.BinnedSumPDF
    zfit.pdf.SplinePDF
    zfit.pdf.UnbinnedFromBinnedPDF
