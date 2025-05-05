Basic PDFs
##########


Basic shapes are fundamendal PDFs, with often well-known functional form.
They are usually fully analytically implemented and often a thin
wrapper around :py:class:`~tensorflow_probability.distribution.Distribution`.
Any missing shape can be easily wrapped using :py:class:`~zfit.pdf.WrapDistribution`.

Below are visualizations of some common PDFs with different parameter values to help
understand their shapes and choose appropriate initial parameter values.

Gaussian PDF
-----------

The Gaussian (or Normal) distribution is characterized by its mean (``mu``) and standard deviation (``sigma``).

.. image:: /images/pdfs/gauss_mu.png
   :width: 80%
   :align: center
   :alt: Gaussian PDF with different mu values

.. image:: /images/pdfs/gauss_sigma.png
   :width: 80%
   :align: center
   :alt: Gaussian PDF with different sigma values

.. autosummary::

    zfit.pdf.Gauss

Exponential PDF
--------------

The Exponential distribution is characterized by its decay parameter (``lambda``).

.. image:: /images/pdfs/exponential_lambda.png
   :width: 80%
   :align: center
   :alt: Exponential PDF with different lambda values

.. autosummary::

    zfit.pdf.Exponential

Uniform PDF
----------

The Uniform distribution is characterized by its lower and upper bounds.

.. image:: /images/pdfs/uniform_range.png
   :width: 80%
   :align: center
   :alt: Uniform PDF with different ranges

.. autosummary::

    zfit.pdf.Uniform


Cauchy PDF
---------

The Cauchy distribution is characterized by its location parameter (``m``) and scale parameter (``gamma``).

.. image:: /images/pdfs/cauchy_m.png
   :width: 80%
   :align: center
   :alt: Cauchy PDF with different m values

.. image:: /images/pdfs/cauchy_gamma.png
   :width: 80%
   :align: center
   :alt: Cauchy PDF with different gamma values

.. autosummary::

    zfit.pdf.Cauchy

Voigt PDF
--------

The Voigt profile is a convolution of a Gaussian and a Lorentzian distribution.

.. image:: /images/pdfs/voigt_sigma.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different sigma values

.. image:: /images/pdfs/voigt_gamma.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different gamma values

.. image:: /images/pdfs/voigt_u.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different u values

.. autosummary::

    zfit.pdf.Voigt

CrystalBall PDF
-------------

The Crystal Ball function is a Gaussian with a power-law tail.

.. image:: /images/pdfs/crystalball_alpha.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different alpha values

.. image:: /images/pdfs/crystalball_n.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different n values

.. image:: /images/pdfs/crystalball_mu.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different mu values

.. image:: /images/pdfs/crystalball_sigma.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different sigma values

.. autosummary::

    zfit.pdf.CrystalBall

LogNormal PDF
-----------

The LogNormal distribution is the distribution of a random variable whose logarithm follows a normal distribution.

.. image:: /images/pdfs/lognormal_mu.png
   :width: 80%
   :align: center
   :alt: LogNormal PDF with different mu values

.. image:: /images/pdfs/lognormal_sigma.png
   :width: 80%
   :align: center
   :alt: LogNormal PDF with different sigma values

... autosummary::

    zfit.pdf.LogNormal
ChiSquared PDF
------------

The Chi-squared distribution is the distribution of a sum of the squares of k independent standard normal random variables.

.. image:: /images/pdfs/chisquared_ndof.png
   :width: 80%
   :align: center
   :alt: ChiSquared PDF with different ndof values

.. autosummary::

    zfit.pdf.ChiSquared

StudentT PDF
----------

The Student's t-distribution is a continuous probability distribution that generalizes the normal distribution.

.. image:: /images/pdfs/studentt_ndof.png
   :width: 80%
   :align: center
   :alt: StudentT PDF with different ndof values

.. autosummary::

    zfit.pdf.StudentT

Gamma PDF
-------

The Gamma distribution is a two-parameter family of continuous probability distributions.

.. image:: /images/pdfs/gamma_gamma.png
   :width: 80%
   :align: center
   :alt: Gamma PDF with different gamma values

.. image:: /images/pdfs/gamma_beta.png
   :width: 80%
   :align: center
   :alt: Gamma PDF with different beta values

.. autosummary::

    zfit.pdf.Gamma

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
    zfit.pdf.GeneralizedGauss
