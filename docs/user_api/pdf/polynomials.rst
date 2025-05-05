Polynomials
#############

While polynomials are also basic PDFs, they convey mathematically
a more special class of functions.

They constitute a sum of different degrees.
Polynomial PDFs are useful for modeling smooth
backgrounds or for creating flexible parametric shapes.

Below are visualizations of polynomial PDFs with different parameter values to help
understand their shapes and choose appropriate initial parameter values.

The general parameter structure includes the `coeff`, which is a list of coefficients starting at the SECOND coefficient. Since the PDF is normalized, the first coefficient is set to constant 1.0 by default; this *can* be changed using the `coeff0` parameter -- it is rarely if ever needed.

To change the overall normalization, use the `extended` parameter.

Bernstein Polynomials
--------------------

Bernstein polynomials are a basis for the space of polynomials defined on the interval [0, 1].
They are particularly useful for modeling smooth shapes with good numerical stability.

.. image:: /images/pdfs/bernstein_degree.png
   :width: 80%
   :align: center
   :alt: Bernstein PDF with different degrees

.. image:: /images/pdfs/bernstein_patterns.png
   :width: 80%
   :align: center
   :alt: Bernstein PDF with different coefficient patterns

Chebyshev Polynomials
-------------------

Chebyshev polynomials are a sequence of orthogonal polynomials defined on the interval [-1, 1].
They are particularly useful for approximating functions with minimal maximum error.

.. image:: /images/pdfs/chebyshev_degree.png
   :width: 80%
   :align: center
   :alt: Chebyshev PDF with different degrees

.. image:: /images/pdfs/chebyshev_patterns.png
   :width: 80%
   :align: center
   :alt: Chebyshev PDF with different coefficient patterns

Legendre Polynomials
-----------------

Legendre polynomials are a sequence of orthogonal polynomials defined on the interval [-1, 1].
They are often used in physics for solving differential equations.

.. image:: /images/pdfs/legendre_degree.png
   :width: 80%
   :align: center
   :alt: Legendre PDF with different degrees

.. image:: /images/pdfs/legendre_patterns.png
   :width: 80%
   :align: center
   :alt: Legendre PDF with different coefficient patterns

.. autosummary::
    :toctree: _generated/polynomials

    zfit.pdf.Bernstein
    zfit.pdf.Chebyshev
    zfit.pdf.Legendre
    zfit.pdf.Chebyshev2
    zfit.pdf.Hermite
    zfit.pdf.Laguerre
    zfit.pdf.RecursivePolynomial
