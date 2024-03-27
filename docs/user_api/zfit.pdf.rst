PDF
---

Basic PDFs
##########

Basic shapes are fundamental PDFs, with often well-known functional form.
They are usually fully analytically implemented and often a thin
wrapper around :py:class:`~tensorflow_probability.distribution.Distribution`.
Any missing shape can be easily wrapped using :py:class:`~zfit.pdf.WrapDistribution`.

.. toctree::
    :maxdepth: 2

    pdf/basic


Binned PDFs
###########

Binned PDFs extend the functionality of unbinned PDFs by providing more histogram-like features in
addition to the basic unbinned PDFs. They interface with the
`Unified Histogram Interface (uhi) <https://uhi.readthedocs.io/en/latest/?badge=latest>`_
that is provided `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/>`_ and especially
`Hist <https://github.com/scikit-hep/hist>`_.

.. toctree::
    :maxdepth: 2

    pdf/binned_pdf


Polynomials
#############

While polynomials are also basic PDFs, they convey mathematically
a more special class of functions.

They constitute a sum of different degrees.

.. toctree::
    :maxdepth: 2

    pdf/polynomials

Kernel Density Estimations
#############################

KDEs provide a means of non-parametric density estimation.

An extensive introduction and explanation can be found in
:ref:`sec-kernel-density-estimation`.

.. toctree::
    :maxdepth: 2

    pdf/kde_api

Composed PDFs
#############################

Composed PDFs build on top of others and provide sums, products and more.

.. toctree::
    :maxdepth: 2

    pdf/composed_pdf

Custom base class
#############################

These base classes are used internally to build PDFs and can also be
used to implement custom PDFs.

They offer more or less support and freedom.

.. toctree::
    :maxdepth: 2

    pdf/custom_base
