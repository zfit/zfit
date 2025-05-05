Composed PDFs
#############################

Composed PDFs build on top of others and provide sums, products and more.
These PDFs allow for creating complex models by combining simpler components.

Below are the available composed PDFs with descriptions of their functionality:

Sum PDF
------

The :py:class:`~zfit.pdf.SumPDF` allows combining multiple PDFs with different fractions.
This is useful for creating mixture models, such as signal plus background.

Product PDF
---------

The :py:class:`~zfit.pdf.ProductPDF` multiplies PDFs together, useful for creating joint distributions.
This is commonly used when variables are independent or when creating multi-dimensional models.

FFT Convolution PDF
----------------

The :py:class:`~zfit.pdf.FFTConvPDFV1` performs convolution of PDFs using Fast Fourier Transform.
This is useful for modeling detector resolution effects or other convolution operations.

Conditional PDF
------------

The :py:class:`~zfit.pdf.ConditionalPDFV1` creates conditional probability distributions.
This allows for modeling dependencies between variables.

Truncated PDF
-----------

The :py:class:`~zfit.pdf.TruncatedPDF` restricts a PDF to a specific range.
This is useful when you need to limit the domain of a PDF without changing its shape within that domain.

.. autosummary::
    :toctree: _generated/composed_pdf

    zfit.pdf.ProductPDF
    zfit.pdf.SumPDF
    zfit.pdf.FFTConvPDFV1
    zfit.pdf.ConditionalPDFV1
    zfit.pdf.TruncatedPDF
