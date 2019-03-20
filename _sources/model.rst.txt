Building a model
================

In order to build a generic model the concept of function and distributed density functions (PDFs) need to be clarified. 
The PDF, or density of a continuous random variable, of X is a function f(x) that describes the relative likelihood for this random variable to take on a given value. 
In this sense, for any two numbers a and b with :math:`a \leq b`, 

:math:`P(a \leq X \leq b) = \int^{b}_{a}f(X)dx`

That is, the probability that X takes on a value in the interval :math:`[a, b]` is the area above this interval and under the graph of the density function.
In other words, in order to a function to be a PDF it must satisfy two criteria: 
(1) :math:`f(x) \neq 0` for all x; (2) :math:`\int^{\infty}_{-\infty}f(x)dx =` are under the entire graph of :math:`f(x)=1`. 
In ``zfit`` these distinctions are respect, i.e. a function can be converted into a PDF by imposing the basic two criteria above. 

Predefined PDFs and basic properties 
''''''''''''''''''''''''''''''''''''

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

In this case the CB object corresponds to a normalised PDF. The main properties of a PDF, e.g. the probability for a given normalisation range or even 
to set a temporary normalisation range can be given as 

.. code-block:: python

    # Get the probabilities of some random generated events
    probs = CB.prob(x=np.random.random(10), norm_range=(5100, 5400))

    # Performing evaluation within a given range 
    with CB.temp_norm_range((5000, 6000)):
        CB.prob(data)  # norm_range is now set

Another feature for the PDF is to calculate its integral in a certain limit. This can be easily achieved by 

.. code-block:: python

    # Calculate the integral between 5230 and 5230 over the PDF normalized
    integral_norm = CB.integrate(limits=(5230, 5230))

In this case the CB has been normalised using the range defined in the observable. 
Conversely, the ``norm_range`` in which the PDF is normalised can also be specified as input. 

Composite PDF
'''''''''''''

A common feature in building composite models it the ability to combine in terms of sum and products different PDFs. 
There are two ways to create such models, either with the class API or with simple Python syntax. 
Let's consider a second crystal ball with the same mean position and width, but different tail parameters

.. code-block:: python

    # New tail parameters for the second CB
    a2     = zfit.Parameter("a2", -1, 0, -10)
    n2     = zfit.Parameter("n2",  1, 0,  10)

    # New crystal Ball function defined in the same observable range
    CB2 = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=a2, n=n2)




Extended PDF
''''''''''''

Custom PDF
''''''''''


Sampling from a PDF
'''''''''''''''''''





