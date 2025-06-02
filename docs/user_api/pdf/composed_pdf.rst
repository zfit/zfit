Composed PDFs
#############################

Composed PDFs build on top of others and provide sums, products and more.
These PDFs allow for creating complex models by combining simpler components.

Below are the available composed PDFs with descriptions of their functionality:

Sum PDF
----------

The :py:class:`~zfit.pdf.SumPDF` allows combining multiple PDFs with different fractions.
This is useful for creating mixture models, such as signal plus background.

.. image:: ../../images/_generated/pdfs/sumpdf_fractions.png
   :width: 80%
   :align: center
   :alt: SumPDF with different component fractions

.. autosummary::

    zfit.pdf.SumPDF

Product PDF
-------------------

The :py:class:`~zfit.pdf.ProductPDF` multiplies PDFs together, useful for creating joint distributions or in the same dimension.
This is commonly used when variables are independent or when creating multi-dimensional models.

.. image:: ../../images/_generated/pdfs/productpdf_1d_multiplication.png
   :width: 80%
   :align: center
   :alt: ProductPDF: Multiplying PDFs in the same dimension

.. image:: ../../images/_generated/pdfs/productpdf_2d_gaussian.png
   :width: 80%
   :align: center
   :alt: ProductPDF: 2D Gaussian

.. image:: ../../images/_generated/pdfs/productpdf_asymmetric.png
   :width: 80%
   :align: center
   :alt: ProductPDF: Asymmetric 2D Gaussian

.. autosummary::

    zfit.pdf.ProductPDF

FFT Convolution PDF
-------------------------------------------------

The :py:class:`~zfit.pdf.FFTConvPDFV1` performs convolution of PDFs using Fast Fourier Transform.
This is useful for modeling detector resolution effects or other convolution operations.

.. image:: ../../images/_generated/pdfs/fftconvpdf_resolutions.png
   :width: 80%
   :align: center
   :alt: FFTConvPDFV1: Gaussian convolved with different resolutions

.. image:: ../../images/_generated/pdfs/fftconvpdf_signals.png
   :width: 80%
   :align: center
   :alt: FFTConvPDFV1: Different signals convolved with Gaussian

.. autosummary::

    zfit.pdf.FFTConvPDFV1

Conditional PDF
---------------------------------------------

The :py:class:`~zfit.pdf.ConditionalPDFV1` creates conditional probability distributions.
This allows for modeling dependencies between variables.

.. image:: ../../images/_generated/pdfs/conditionalpdf_gaussian.png
   :width: 80%
   :align: center
   :alt: ConditionalPDFV1: Gaussian with mean depending on x

.. image:: ../../images/_generated/pdfs/conditionalpdf_width.png
   :width: 80%
   :align: center
   :alt: ConditionalPDFV1: Gaussian with width depending on x

.. autosummary::

    zfit.pdf.ConditionalPDFV1

Truncated PDF
--------------------------------------------

The :py:class:`~zfit.pdf.TruncatedPDF` restricts a PDF to a specific range.
This is useful when you need to limit the domain of a PDF without changing its shape within that domain.

.. image:: ../../images/_generated/pdfs/truncatedpdf_gaussian.png
   :width: 80%
   :align: center
   :alt: TruncatedPDF: Gaussian with different truncation ranges

.. image:: ../../images/_generated/pdfs/truncatedpdf_various.png
   :width: 80%
   :align: center
   :alt: TruncatedPDF: Different PDFs truncated to [-2, 2]

.. autosummary::

    zfit.pdf.TruncatedPDF

All Composed PDFs

.. autosummary::
    :toctree: _generated/composed_pdf

    zfit.pdf.ProductPDF
    zfit.pdf.SumPDF
    zfit.pdf.FFTConvPDFV1
    zfit.pdf.ConditionalPDFV1
    zfit.pdf.TruncatedPDF
