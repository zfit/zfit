Basic PDFs
##########


Basic shapes are fundamendal PDFs, with often well-known functional form.
They are usually fully analytically implemented and often a thin
wrapper around :py:class:`~tensorflow_probability.distribution.Distribution`.
Any missing shape can be easily wrapped using :py:class:`~zfit.pdf.WrapDistribution`.

Below are visualizations of some common PDFs with different parameter values to help
understand their shapes and choose appropriate initial parameter values.

Gaussian PDF
---------------------------------------------------------------------

The :py:class:`~zfit.pdf.Gauss` (or Normal) distribution is characterized by its mean (``mu``) and standard deviation (``sigma``).

.. image:: ../../images/_generated/pdfs/gauss_mu.png
   :width: 80%
   :align: center
   :alt: Gaussian PDF with different mu values

.. image:: ../../images/_generated/pdfs/gauss_sigma.png
   :width: 80%
   :align: center
   :alt: Gaussian PDF with different sigma values

.. autosummary::

    zfit.pdf.Gauss

Exponential PDF
---------------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.Exponential` distribution is characterized by its decay parameter (``lambda``).

.. image:: ../../images/_generated/pdfs/exponential_lambda.png
   :width: 80%
   :align: center
   :alt: Exponential PDF with different lambda values

.. autosummary::

    zfit.pdf.Exponential

Uniform PDF
--------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.Uniform` distribution is characterized by its lower and upper bounds.

.. image:: ../../images/_generated/pdfs/uniform_range.png
   :width: 80%
   :align: center
   :alt: Uniform PDF with different ranges

.. autosummary::

    zfit.pdf.Uniform


Cauchy PDF
----------------------------------------------------------------

The :py:class:`~zfit.pdf.Cauchy` distribution is characterized by its location parameter (``m``) and scale parameter (``gamma``).

.. image:: ../../images/_generated/pdfs/cauchy_m.png
   :width: 80%
   :align: center
   :alt: Cauchy PDF with different m values

.. image:: ../../images/_generated/pdfs/cauchy_gamma.png
   :width: 80%
   :align: center
   :alt: Cauchy PDF with different gamma values

.. autosummary::

    zfit.pdf.Cauchy

Voigt PDF
---------------------------------------------------------------

The :py:class:`~zfit.pdf.Voigt` profile is a convolution of a Gaussian and a Lorentzian distribution.

.. image:: ../../images/_generated/pdfs/voigt_sigma.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different sigma values

.. image:: ../../images/_generated/pdfs/voigt_gamma.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different gamma values

.. image:: ../../images/_generated/pdfs/voigt_m.png
   :width: 80%
   :align: center
   :alt: Voigt PDF with different u values

.. autosummary::

    zfit.pdf.Voigt

CrystalBall PDF
--------------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.CrystalBall` function is a Gaussian with a power-law tail.

.. image:: ../../images/_generated/pdfs/crystalball_alpha.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different alpha values

.. image:: ../../images/_generated/pdfs/crystalball_n.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different n values

.. image:: ../../images/_generated/pdfs/crystalball_mu.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different mu values

.. image:: ../../images/_generated/pdfs/crystalball_sigma.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different sigma values

.. autosummary::

    zfit.pdf.CrystalBall

LogNormal PDF
---------------------------------------------------------------------

The :py:class:`~zfit.pdf.LogNormal` distribution is the distribution of a random variable whose logarithm follows a normal distribution.

.. image:: ../../images/_generated/pdfs/lognormal_mu.png
   :width: 80%
   :align: center
   :alt: LogNormal PDF with different mu values

.. image:: ../../images/_generated/pdfs/lognormal_sigma.png
   :width: 80%
   :align: center
   :alt: LogNormal PDF with different sigma values

.. autosummary::

    zfit.pdf.LogNormal

ChiSquared PDF
--------------------------------------------
The :py:class:`~zfit.pdf.ChiSquared` distribution is the distribution of a sum of the squares of k independent standard normal random variables.

.. image:: ../../images/_generated/pdfs/chisquared_ndof.png
   :width: 80%
   :align: center
   :alt: ChiSquared PDF with different ndof values

.. autosummary::

    zfit.pdf.ChiSquared

StudentT PDF
--------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.StudentT` t-distribution is a continuous probability distribution that generalizes the normal distribution.

.. image:: ../../images/_generated/pdfs/studentt_ndof.png
   :width: 80%
   :align: center
   :alt: StudentT PDF with different ndof values

.. autosummary::

    zfit.pdf.StudentT

Gamma PDF
-----------------------------------------------

The :py:class:`~zfit.pdf.Gamma` distribution is a two-parameter family of continuous probability distributions.

.. image:: ../../images/_generated/pdfs/gamma_gamma.png
   :width: 80%
   :align: center
   :alt: Gamma PDF with different gamma values

.. image:: ../../images/_generated/pdfs/gamma_beta.png
   :width: 80%
   :align: center
   :alt: Gamma PDF with different beta values

.. autosummary::

    zfit.pdf.Gamma

BifurGauss PDF
---------------------------------------------------------------------

The :py:class:`~zfit.pdf.BifurGauss` distribution is a Gaussian with different widths on the left and right sides.

.. image:: ../../images/_generated/pdfs/bifurgauss_mu.png
   :width: 80%
   :align: center
   :alt: BifurGauss PDF with different mu values

.. image:: ../../images/_generated/pdfs/bifurgauss_sigmal.png
   :width: 80%
   :align: center
   :alt: BifurGauss PDF with different sigma_left values

.. image:: ../../images/_generated/pdfs/bifurgauss_sigmar.png
   :width: 80%
   :align: center
   :alt: BifurGauss PDF with different sigma_right values

.. autosummary::

    zfit.pdf.BifurGauss

Poisson PDF
---------------------------------------------------------------

The :py:class:`~zfit.pdf.Poisson` distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space.

.. image:: ../../images/_generated/pdfs/poisson_lambda.png
   :width: 80%
   :align: center
   :alt: Poisson PDF with different lambda values

.. autosummary::

    zfit.pdf.Poisson

QGauss PDF
-----------------------------------------------

The :py:class:`~zfit.pdf.QGauss` distribution is a q-Gaussian distribution, which is a generalization of the normal distribution.

.. image:: ../../images/_generated/pdfs/qgauss_mu.png
   :width: 80%
   :align: center
   :alt: QGauss PDF with different mu values

.. image:: ../../images/_generated/pdfs/qgauss_sigma.png
   :width: 80%
   :align: center
   :alt: QGauss PDF with different sigma values

.. image:: ../../images/_generated/pdfs/qgauss_q.png
   :width: 80%
   :align: center
   :alt: QGauss PDF with different q values

.. autosummary::

    zfit.pdf.QGauss

JohnsonSU PDF
--------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.JohnsonSU` distribution is a four-parameter family of probability distributions.

.. image:: ../../images/_generated/pdfs/johnsonsu_mu.png
   :width: 80%
   :align: center
   :alt: JohnsonSU PDF with different mu values

.. image:: ../../images/_generated/pdfs/johnsonsu_gamma.png
   :width: 80%
   :align: center
   :alt: JohnsonSU PDF with different gamma values

.. image:: ../../images/_generated/pdfs/johnsonsu_delta.png
   :width: 80%
   :align: center
   :alt: JohnsonSU PDF with different delta values

.. autosummary::

    zfit.pdf.JohnsonSU

GeneralizedGauss PDF
-----------------------------------------------------------

The :py:class:`~zfit.pdf.GeneralizedGauss` distribution is a generalization of the normal distribution with an additional shape parameter.

.. image:: ../../images/_generated/pdfs/generalizedgauss_mu.png
   :width: 80%
   :align: center
   :alt: GeneralizedGauss PDF with different mu values

.. image:: ../../images/_generated/pdfs/generalizedgauss_sigma.png
   :width: 80%
   :align: center
   :alt: GeneralizedGauss PDF with different sigma values

.. image:: ../../images/_generated/pdfs/generalizedgauss_beta.png
   :width: 80%
   :align: center
   :alt: GeneralizedGauss PDF with different beta values

.. autosummary::

    zfit.pdf.GeneralizedGauss

TruncatedGauss PDF
---------------------------------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.TruncatedGauss` distribution is a Gaussian distribution that is truncated to a specified range.

.. image:: ../../images/_generated/pdfs/truncatedgauss_mu.png
   :width: 80%
   :align: center
   :alt: TruncatedGauss PDF with different mu values

.. image:: ../../images/_generated/pdfs/truncatedgauss_sigma.png
   :width: 80%
   :align: center
   :alt: TruncatedGauss PDF with different sigma values

.. image:: ../../images/_generated/pdfs/truncatedgauss_range.png
   :width: 80%
   :align: center
   :alt: TruncatedGauss PDF with different truncation ranges

.. autosummary::

    zfit.pdf.TruncatedGauss

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
