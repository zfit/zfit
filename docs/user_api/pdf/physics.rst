Physics PDFs
##############

Physics PDFs are PDFs that are often used in high energy physics. They are in the
zfit-physics package, which needs to be installed separately (for example, via
``pip install zfit-physics``) and accessed via the ``zfit_physics.pdf`` module.

Physics PDFs are specialized probability density functions commonly used in particle physics
for modeling signal and background distributions. These PDFs are designed to capture specific
physical phenomena observed in experimental data.

Below are visualizations of physics PDFs with different parameter values to help
understand their shapes and choose appropriate initial parameter values.

CrystalBall PDF
-------------

The :py:class:`~zfit.pdf.CrystalBall` function is a Gaussian with a power-law tail, commonly used to model
energy loss and detector resolution effects in particle physics.

.. image:: _generated/pdfs/crystalball_alpha.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different alpha values

.. image:: _generated/pdfs/crystalball_n.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different n values

.. image:: _generated/pdfs/crystalball_mu.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different mu values

.. image:: _generated/pdfs/crystalball_sigma.png
   :width: 80%
   :align: center
   :alt: CrystalBall PDF with different sigma values

DoubleCB PDF
---------

The :py:class:`~zfit.pdf.DoubleCB` function extends the Crystal Ball by having power-law tails on both sides
of the Gaussian core, providing more flexibility for modeling asymmetric peaks.

.. image:: _generated/pdfs/doublecb_alphal.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaL values

.. image:: _generated/pdfs/doublecb_alphar.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaR values

GaussExpTail PDF
-------------

The :py:class:`~zfit.pdf.GaussExpTail` combines a Gaussian core with an exponential tail,
useful for modeling detector resolution effects with asymmetric tails.

.. image:: _generated/pdfs/gaussexptail_alpha.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different alpha values

.. image:: _generated/pdfs/gaussexptail_sigma.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different sigma values

GeneralizedCB PDF
--------------

The :py:class:`~zfit.pdf.GeneralizedCB` extends the Crystal Ball function with additional parameters
for more flexible modeling of asymmetric peaks with power-law tails.

.. image:: _generated/pdfs/generalizedcb_alpha.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different alpha values

.. image:: _generated/pdfs/generalizedcb_n.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different n values

.. image:: _generated/pdfs/generalizedcb_t.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different t values

GeneralizedGaussExpTail PDF
------------------------

The :py:class:`~zfit.pdf.GeneralizedGaussExpTail` extends the GaussExpTail function with additional
parameters for more flexible modeling of asymmetric distributions.

.. image:: _generated/pdfs/generalizedgaussexptail_alpha.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different alpha values

.. image:: _generated/pdfs/generalizedgaussexptail_k.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different k values

.. image:: _generated/pdfs/generalizedgaussexptail_sigma.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different sigma values

.. autosummary::
    :toctree: _generated/physics

    zfit.pdf.CrystalBall
    zfit.pdf.DoubleCB
    zfit.pdf.GeneralizedCB
    zfit.pdf.GaussExpTail
    zfit.pdf.GeneralizedGaussExpTail

External Physics PDFs
------------------

The following PDFs are available in the separate `zfit_physics` package:

.. autosummary::
    :toctree: _generated/physics_external

    zfit_physics.pdf.Argus
    zfit_physics.pdf.RelativisticBreitWigner
    zfit_physics.pdf.CMSShape
    zfit_physics.pdf.Cruijff
    zfit_physics.pdf.ErfExp
    zfit_physics.pdf.Novosibirsk
    zfit_physics.pdf.Tsallis
