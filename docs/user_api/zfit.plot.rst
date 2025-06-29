Plot
------------

.. warning::
    This module is mainly intended for interactive usage, it is experimental and still in development. The API might change without warning and breaks code.


Utilities to plot PDFs and functions. These are simple helpers to plot models and their components, 
they are not meant to be a full-fledged plotting library but provide a quick way to visualize PDFs.

The plotting is based on ``matplotlib`` and integrates closely with it.

Quick Start
^^^^^^^^^^^

The easiest way to plot a PDF is using the ``.plot`` attribute available on all PDFs::

    # Plot a simple PDF
    pdf.plot.plotpdf()
    
    # Plot an extended PDF (scaled by yield)
    pdf.plot.plotpdf(extended=True)
    
    # Plot with custom styling
    pdf.plot.plotpdf(linestyle='--', color='red', label='My PDF')
    
    # For composite PDFs (like SumPDF), plot components
    sumpdf.plot.comp.plotpdf()

Examples
^^^^^^^^

**Basic plotting**::

    import zfit
    import matplotlib.pyplot as plt
    
    # Create a simple Gaussian
    obs = zfit.Space("x", limits=(-5, 5))
    gauss = zfit.pdf.Gauss(mu=0, sigma=1, obs=obs)
    
    # Plot the PDF
    gauss.plot.plotpdf()
    plt.show()

**Plotting extended PDFs with components**::

    # Create extended PDFs
    gauss1 = zfit.pdf.Gauss(mu=-1, sigma=0.5, obs=obs).create_extended(100)
    gauss2 = zfit.pdf.Gauss(mu=1, sigma=0.8, obs=obs).create_extended(150)
    
    # Create a SumPDF (automatically extended)
    sumpdf = zfit.pdf.SumPDF([gauss1, gauss2])
    
    # Plot the sum and its components
    sumpdf.plot.plotpdf(label='Sum')
    sumpdf.plot.comp.plotpdf(linestyle='--')
    plt.legend()
    plt.show()

**Custom plotting with data overlay**::

    # Generate and plot data
    data = sumpdf.sample(1000)
    plt.hist(data.value(), bins=50, density=True, alpha=0.5, label='Data')
    
    # Overlay the PDF
    sumpdf.plot.plotpdf(extended=False, label='PDF')
    plt.legend()
    plt.show()

Available Functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _generated/plot

    zfit.plot.plot_model_pdfV1
    zfit.plot.plot_sumpdf_components_pdfV1
