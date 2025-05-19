Polynomials
#############

While polynomials are also basic PDFs, they convey mathematically
a more special class of functions.

They constitute a sum of different degrees.
Polynomial PDFs are useful for modeling smooth
backgrounds or for creating flexible parametric shapes.

Below are visualizations of polynomial PDFs with different parameter values to help
understand their shapes and choose appropriate initial parameter values.

The general parameter structure includes the ``coeff``, which is a list of coefficients starting at the SECOND coefficient. Since the PDF is normalized, the first coefficient is set to constant 1.0 by default; this *can* be changed using the ``coeff0`` parameter -- it is rarely if ever needed.

To change the overall normalization, use the ``extended`` parameter.

Bernstein Polynomials
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

:py:class:`~zfit.pdf.Bernstein` polynomials are a basis for the space of polynomials defined on the interval [0, 1].
They are particularly useful for modeling smooth shapes with good numerical stability.

.. image:: ../../images/_generated/pdfs/bernstein_degree.png
   :width: 80%
   :align: center
   :alt: Bernstein PDF with different degrees

.. image:: ../../images/_generated/pdfs/bernstein_patterns.png
   :width: 80%
   :align: center
   :alt: Bernstein PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Bernstein

Chebyshev Polynomials
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

:py:class:`~zfit.pdf.Chebyshev` polynomials are a sequence of orthogonal polynomials defined on the interval [-1, 1].
They are particularly useful for approximating functions with minimal maximum error.

.. image:: ../../images/_generated/pdfs/chebyshev_degree.png
   :width: 80%
   :align: center
   :alt: Chebyshev PDF with different degrees

.. image:: ../../images/_generated/pdfs/chebyshev_patterns.png
   :width: 80%
   :align: center
   :alt: Chebyshev PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Chebyshev

Legendre Polynomials
--------------------------------------------------

:py:class:`~zfit.pdf.Legendre` polynomials are a sequence of orthogonal polynomials defined on the interval [-1, 1].
They are often used in physics for solving differential equations.

.. image:: ../../images/_generated/pdfs/legendre_degree.png
   :width: 80%
   :align: center
   :alt: Legendre PDF with different degrees

.. image:: ../../images/_generated/pdfs/legendre_patterns.png
   :width: 80%
   :align: center
   :alt: Legendre PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Legendre

Chebyshev2 Polynomials
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

:py:class:`~zfit.pdf.Chebyshev2` polynomials are a sequence of orthogonal polynomials of the second kind defined on the interval [-1, 1].

.. image:: ../../images/_generated/pdfs/chebyshev2_degree.png
   :width: 80%
   :align: center
   :alt: Chebyshev2 PDF with different degrees

.. image:: ../../images/_generated/pdfs/chebyshev2_patterns.png
   :width: 80%
   :align: center
   :alt: Chebyshev2 PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Chebyshev2

Hermite Polynomials
------------------------------------------------

:py:class:`~zfit.pdf.Hermite` polynomials are a sequence of orthogonal polynomials that arise in probability, quantum mechanics, and other fields.

.. image:: ../../images/_generated/pdfs/hermite_degree.png
   :width: 80%
   :align: center
   :alt: Hermite PDF with different degrees

.. image:: ../../images/_generated/pdfs/hermite_patterns.png
   :width: 80%
   :align: center
   :alt: Hermite PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Hermite

Laguerre Polynomials
-------------------------------------------------

:py:class:`~zfit.pdf.Laguerre` polynomials are a sequence of orthogonal polynomials associated with the Gamma distribution.

.. image:: ../../images/_generated/pdfs/laguerre_degree.png
   :width: 80%
   :align: center
   :alt: Laguerre PDF with different degrees

.. image:: ../../images/_generated/pdfs/laguerre_patterns.png
   :width: 80%
   :align: center
   :alt: Laguerre PDF with different coefficient patterns

.. autosummary::

    zfit.pdf.Laguerre

RecursivePolynomial
-------------------------------------------------

:py:class:`~zfit.pdf.RecursivePolynomial` provides a general framework for defining polynomials through recursive relations.

.. autosummary::
    :toctree: _generated/polynomials

    zfit.pdf.Bernstein
    zfit.pdf.Chebyshev
    zfit.pdf.Legendre
    zfit.pdf.Chebyshev2
    zfit.pdf.Hermite
    zfit.pdf.Laguerre
    zfit.pdf.RecursivePolynomial
