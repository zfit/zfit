Kernel Density Estimations
#############################

KDEs (Kernel Density Estimations) provide a means of non-parametric density estimation. They are useful when the underlying distribution of data is unknown or complex, and a parametric model might not be appropriate.

An extensive introduction and explanation can be found in
:ref:`sec-kernel-density-estimation`.

Below are visualizations of KDEs with different parameter values to help understand their behavior and choose appropriate settings.

Bandwidth Effect
--------------------------------------------------------------------------------------------

The bandwidth parameter controls the smoothness of the KDE. A smaller bandwidth captures more detail but might overfit to noise, while a larger bandwidth produces a smoother estimate but might miss important features.

.. image:: ../../images/_generated/pdfs/kde_bandwidth.png
   :width: 80%
   :align: center
   :alt: KDE with different bandwidth values

Kernel Types
----------------------------------------------------------------------------------------

Different kernel functions can be used in KDEs. The default is a Gaussian kernel, but other distributions like Student's T can be used for different tail behaviors.

.. image:: ../../images/_generated/pdfs/kde_kernel.png
   :width: 80%
   :align: center
   :alt: KDE with different kernel types

KDE Implementations
---------------------------------------------------------------------------------------------

zfit provides several KDE implementations with different trade-offs between accuracy and computational efficiency:

- **KDE1DimExact**: Calculates the true sum of kernels (most accurate but slower for large datasets)
- **KDE1DimGrid**: Uses a binning approach (faster for large datasets)
- **KDE1DimFFT**: Uses Fast Fourier Transform for even faster computation
- **KDE1DimISJ**: Uses the Improved Sheather-Jones algorithm for optimal bandwidth selection

.. image:: ../../images/_generated/pdfs/kde_implementations.png
   :width: 80%
   :align: center
   :alt: Different KDE implementations

.. autosummary::
    :toctree: _generated/kde_api

    zfit.pdf.KDE1DimExact
    zfit.pdf.KDE1DimGrid
    zfit.pdf.KDE1DimFFT
    zfit.pdf.KDE1DimISJ
    zfit.pdf.GaussianKDE1DimV1
