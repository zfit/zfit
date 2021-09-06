PDF
---

.. todo:: Fill overview page

Basic PDFs
##########

Basic shapes are fundamendal PDFs, with often well-known functional form.
They are usually fully analytically implemented and often a thin
wrapper around :py:class:`~tensorflow_probability.distribution.Distribution`.
Any missing shape can be easily wrapped using :py:class:``~zfit.pdf.WrapDistribution``.

.. toctree::
    :maxdepth: 2

    pdf/basic

Polynomials
#############

While polynomials are also basic PDFs, they convey mathematically
a more special class of functions.

They constitute a sum of different degrees.

.. toctree::
    :maxdepth: 2

    pdf/polynomials

Kernel Density Estimtations
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
