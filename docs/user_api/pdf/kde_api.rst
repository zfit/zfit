Kernel Density Estimations
#############################

KDEs (Kernel Density Estimations) provide a means of non-parametric density estimation. They are useful when the underlying distribution of data is unknown or complex, and a parametric model might not be appropriate.

Below are visualizations of KDEs with different parameters to help understand their behavior.

KDE with Different Bandwidth Values
---------------------------------------------------------------------

The bandwidth parameter controls the smoothness of the density estimate. Smaller values capture more detail but may overfit, while larger values produce smoother estimates but may miss important features.

.. image:: ../../images/_generated/pdfs/kde_bandwidth.png
   :width: 80%
   :align: center
   :alt: KDE with different bandwidth values

Different KDE Implementations
---------------------------------------------------------------------

zfit provides several KDE implementations with different performance characteristics.

.. image:: ../../images/_generated/pdfs/kde_implementations.png
   :width: 80%
   :align: center
   :alt: Different KDE implementations

KDE with Different Bandwidth Methods
---------------------------------------------------------------------

Various methods exist for automatically determining the optimal bandwidth.

.. image:: ../../images/_generated/pdfs/kde_bandwidth_methods.png
   :width: 80%
   :align: center
   :alt: KDE with different bandwidth methods

KDE1DimISJ Implementation
---------------------------------------------------------------------

The ISJ (Improved Sheather-Jones) method provides an adaptive bandwidth selection.

.. image:: ../../images/_generated/pdfs/kde_isj.png
   :width: 80%
   :align: center
   :alt: KDE1DimISJ implementation

KDE with Different Kernel Types
---------------------------------------------------------------------

The kernel function defines the shape of the influence of each data point. Different kernel functions can produce different density estimates.

.. image:: ../../images/_generated/pdfs/kde_kernel.png
   :width: 80%
   :align: center
   :alt: KDE with different kernel types

An extensive introduction and explanation can be found in
:ref:`sec-kernel-density-estimation`.

.. autosummary::
   :toctree: _generated/kde

   zfit.pdf.KDE1DimExact
   zfit.pdf.KDE1DimGrid
   zfit.pdf.KDE1DimFFT
   zfit.pdf.KDE1DimISJ
   zfit.pdf.GaussianKDE1DimV1
