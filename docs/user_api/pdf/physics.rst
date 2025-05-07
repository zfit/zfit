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
----------------------------------------------

The :py:class:`~zfit.pdf.CrystalBall` function is a Gaussian with a power-law tail, commonly used to model
energy loss and detector resolution effects in particle physics.

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


GaussExpTail PDF
----------------------------------------------

The :py:class:`~zfit.pdf.GaussExpTail` combines a Gaussian core with an exponential tail,
useful for modeling detector resolution effects with asymmetric tails.

.. image:: ../../images/_generated/pdfs/gaussexptail_alpha.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different alpha values

.. image:: ../../images/_generated/pdfs/gaussexptail_sigma.png
   :width: 80%
   :align: center
   :alt: GaussExpTail PDF with different sigma values

.. autosummary::

    zfit.pdf.GaussExpTail

GeneralizedCB PDF
-----------------------------------------------

The :py:class:`~zfit.pdf.GeneralizedCB` extends the Crystal Ball function with additional parameters
for more flexible modeling of asymmetric peaks with power-law tails.

.. image:: ../../images/_generated/pdfs/generalizedcb_alphal.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different alphaL values

.. image:: ../../images/_generated/pdfs/generalizedcb_nl.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different nL values

.. image:: ../../images/_generated/pdfs/generalizedcb_alphar.png
   :width: 80%
   :align: center
   :alt: GeneralizedCB PDF with different alphaR values

.. autosummary::

    zfit.pdf.GeneralizedCB

GeneralizedGaussExpTail PDF
------------------------------------------------------------------------------------------

The :py:class:`~zfit.pdf.GeneralizedGaussExpTail` extends the GaussExpTail function with additional
parameters for more flexible modeling of asymmetric distributions.

.. image:: ../../images/_generated/pdfs/generalizedgaussexptail_alphal.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different alphaL values

.. image:: ../../images/_generated/pdfs/generalizedgaussexptail_alphar.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different alphaR values

.. image:: ../../images/_generated/pdfs/generalizedgaussexptail_sigmal.png
   :width: 80%
   :align: center
   :alt: GeneralizedGaussExpTail PDF with different sigmaL values

.. autosummary::

    zfit.pdf.GeneralizedGaussExpTail

DoubleCB PDF
---------------------------------------------------------

The :py:class:`~zfit.pdf.DoubleCB` function extends the Crystal Ball by having power-law tails on both sides
of the Gaussian core, providing more flexibility for modeling asymmetric peaks.

.. image:: ../../images/_generated/pdfs/doublecb_alphal.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaL values

.. image:: ../../images/_generated/pdfs/doublecb_alphar.png
   :width: 80%
   :align: center
   :alt: DoubleCB PDF with different alphaR values

.. autosummary::

    zfit.pdf.DoubleCB


The following PDFs are available in the separate ``zfit_physics`` package:


Argus
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.Argus` PDF is commonly used to model background distributions in B physics, particularly for describing the kinematic threshold behavior.

.. image:: ../../images/_generated/pdfs/argus_c.png
   :width: 80%
   :align: center
   :alt: Argus PDF with different c values

.. image:: ../../images/_generated/pdfs/argus_m0.png
   :width: 80%
   :align: center
   :alt: Argus PDF with different m0 values

.. image:: ../../images/_generated/pdfs/argus_p.png
   :width: 80%
   :align: center
   :alt: Argus PDF with different p values

.. autosummary::

    zfit_physics.pdf.Argus

RelativisticBreitWigner
----------------------------------------------------------------------------------------------

The :py:class:`~zfit_physics.pdf.RelativisticBreitWigner` PDF describes the mass distribution of unstable particles, taking into account relativistic effects.

.. image:: ../../images/_generated/pdfs/rbw_m.png
   :width: 80%
   :align: center
   :alt: RelativisticBreitWigner PDF with different m0 values

.. image:: ../../images/_generated/pdfs/rbw_gamma.png
   :width: 80%
   :align: center
   :alt: RelativisticBreitWigner PDF with different gamma values

.. autosummary::

    zfit_physics.pdf.RelativisticBreitWigner

CMSShape
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.CMSShape` PDF is used to model background distributions in CMS analyses.

.. image:: ../../images/_generated/pdfs/cms_m.png
   :width: 80%
   :align: center
   :alt: CMSShape PDF with different alpha values

.. image:: ../../images/_generated/pdfs/cms_beta.png
   :width: 80%
   :align: center
   :alt: CMSShape PDF with different beta values

.. image:: ../../images/_generated/pdfs/cms_gamma.png
   :width: 80%
   :align: center
   :alt: CMSShape PDF with different gamma values

.. autosummary::

    zfit_physics.pdf.CMSShape

Cruijff
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.Cruijff` PDF is an asymmetric Gaussian with different widths and tails on each side, often used to model detector resolution effects.

.. image:: ../../images/_generated/pdfs/cruijff_mu.png
   :width: 80%
   :align: center
   :alt: Cruijff PDF with different mu values

.. image:: ../../images/_generated/pdfs/cruijff_sigmal.png
   :width: 80%
   :align: center
   :alt: Cruijff PDF with different sigmaL values

.. image:: ../../images/_generated/pdfs/cruijff_sigmar.png
   :width: 80%
   :align: center
   :alt: Cruijff PDF with different sigmaR values

.. image:: ../../images/_generated/pdfs/cruijff_alphal.png
   :width: 80%
   :align: center
   :alt: Cruijff PDF with different alphaL values

.. image:: ../../images/_generated/pdfs/cruijff_alphar.png
   :width: 80%
   :align: center
   :alt: Cruijff PDF with different alphaR values

.. autosummary::

    zfit_physics.pdf.Cruijff

ErfExp
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.ErfExp` PDF combines an error function with an exponential, useful for modeling backgrounds with a turn-on effect.

.. image:: ../../images/_generated/pdfs/erfexp_mu.png
   :width: 80%
   :align: center
   :alt: ErfExp PDF with different mu values

.. image:: ../../images/_generated/pdfs/erfexp_beta.png
   :width: 80%
   :align: center
   :alt: ErfExp PDF with different beta values

.. image:: ../../images/_generated/pdfs/erfexp_gamma.png
   :width: 80%
   :align: center
   :alt: ErfExp PDF with different gamma values

.. image:: ../../images/_generated/pdfs/erfexp_n.png
   :width: 80%
   :align: center
   :alt: ErfExp PDF with different n values

.. autosummary::

    zfit_physics.pdf.ErfExp

Novosibirsk
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.Novosibirsk` PDF is used to model asymmetric peaks with a Gaussian-like core and exponential tails.

.. image:: ../../images/_generated/pdfs/novo_mu.png
   :width: 80%
   :align: center
   :alt: Novosibirsk PDF with different mu values

.. image:: ../../images/_generated/pdfs/novo_sigma.png
   :width: 80%
   :align: center
   :alt: Novosibirsk PDF with different sigma values

.. image:: ../../images/_generated/pdfs/novo_lambd.png
   :width: 80%
   :align: center
   :alt: Novosibirsk PDF with different lambd values

.. autosummary::

    zfit_physics.pdf.Novosibirsk

Tsallis
---------------------------------------------------

The :py:class:`~zfit_physics.pdf.Tsallis` PDF is used in high-energy physics to model particle production spectra.

.. image:: ../../images/_generated/pdfs/tsallis_m.png
   :width: 80%
   :align: center
   :alt: Tsallis PDF with different m values

.. image:: ../../images/_generated/pdfs/tsallis_n.png
   :width: 80%
   :align: center
   :alt: Tsallis PDF with different n values

.. image:: ../../images/_generated/pdfs/tsallis_t.png
   :width: 80%
   :align: center
   :alt: Tsallis PDF with different t values

.. autosummary::

    zfit_physics.pdf.Tsallis






.. autosummary::
    :toctree: _generated/physics

    zfit.pdf.CrystalBall
    zfit.pdf.DoubleCB
    zfit.pdf.GeneralizedCB
    zfit.pdf.GaussExpTail
    zfit.pdf.GeneralizedGaussExpTail

.. autosummary::
    :toctree: _generated/physics_external

    zfit_physics.pdf.Argus
    zfit_physics.pdf.RelativisticBreitWigner
    zfit_physics.pdf.CMSShape
    zfit_physics.pdf.Cruijff
    zfit_physics.pdf.ErfExp
    zfit_physics.pdf.Novosibirsk
    zfit_physics.pdf.Tsallis
