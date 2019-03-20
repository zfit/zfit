Building a model
================

In order to build a generic model the concept of function and distributed density functions (PDFs) need to be clarified. 
The PDF, or density of a continuous random variable, of X is a function f(x) that describes the relative likelihood for this random variable to take on a given value. In this sense, for any two numbers a and b with :math:`a \leq b`, 

:math:`P(a \leq X \leq b) = \int^{b}_{a}f(X)dx`

That is, the probability that X takes on a value in the interval [a, b] is the area above this interval and under the graph of the density function.
In other words, in order to a function to be a PDF it must satisfy two criteria: (1) :math:`f(x) \neq 0` for all x; (2) :math:`int^{\infty}_{-\infty}f(x)dx =` are under the entire graph of :math:`f(x)=1`. 
In ``zfit`` these distinctions are respect, i.e. a function can be converted into a PDF by imposing the basic two criteria above. 

Predefined PDFs basic operations
================================

A series of predefined PDFs are available to the users and can be easily accessed using autocompletion (if available). In fact, all of these can also be seen in 

.. code-block:: python

    print(zfit.pdf.__all__)
    ['BasePDF', 'Exponential', 'CrystalBall', 'Gauss', 'Uniform', 'WrapDistribution', 'ProductPDF', 'SumPDF']

These include the basic function but also some operations discussed below. Let's consider the simple example of a ``CrystalBall``.
PDF objects must also be initialised giving their named parameters. For example:

.. code-block:: python

    # Creating the parameters for the crystal ball
    mu    = zfit.Parameter("mu"   , 5279, 5100, 5300)
    sigma = zfit.Parameter("sigma",   20, 0, 50)
    a     = zfit.Parameter("a"    ,    1, 0, 10)
    n     = zfit.Parameter("n"    ,    1, 0, 10)

    # Single crystal Ball
    CB = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a, n=n)





Extended PDF
============

Composite PDF
=============

Custom PDF
==========


Sampling from a PDF
===================





