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

The Crystal Ball function is a Gaussian with a power-law tail, commonly used to model
energy loss and detector resolution effects in particle physics.

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

DoubleCB PDF
---------

The Double Crystal Ball function extends the Crystal Ball by having power-law tails on both sides
of the Gaussian core, providing more flexibility for modeling asymmetric peaks.

.. image:: /images/pdfs/doublecb_alphal.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaL values

.. image:: /images/pdfs/doublecb_alphar.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaR values

GaussExpTail PDF
-------------

The Gaussian with Exponential Tail combines a Gaussian core with an exponential tail,
useful for modeling detector resolution effects with asymmetric tails.

.. image:: /images/pdfs/gaussexptail_alpha.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different alpha values

.. image:: /images/pdfs/gaussexptail_sigma.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different sigma values

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
