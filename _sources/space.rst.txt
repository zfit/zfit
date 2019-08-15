============================
Space, Observable and Range
============================

Inside zfit, :py:class:`~zfit.Space` defines the domain of objects by specifying the observables/axes and *maybe* also
the limits. Any model and data needs to be specified in a certain domain, which is usually done using the
``obs`` argument. It is crucial that the axis used by the observable of the data and the model match, and this matching is
handle by the :py:class:`~zfit.Space` class.

.. code:: python

    obs = zfit.Space("x")
    model = zfit.pdf.Gauss(obs=obs, ...)
    data = zfit.Data.from_numpy(obs=obs, ...)

Definitions
-----------
**Space**: an *n*-dimensional definition of a domain (either by using one or more observables or axes),
with or without limits.

.. note::

    *compared to `RooFit`, a space is **not** the equivalent of an observable but rather corresponds
    to an object combining **a set** of observables (which of course can be of size 1). Furthermore,
    there is a **strong** distinction in zfit between a :py:class:`~zfit.Space` (or observables)
    and a :py:class:`~zfit.Parameter`, both conceptually and in terms of implementation and usage.*

**Observable**: a string defining the axes; a named axes.

*(for advanced usage only, can be skipped on first read)*
**Axis**: integer defining the axes *internally* of a model. There is always a mapping of observables <-> axes *once inside a model*.

**Limit** The range on a certain axis. Typically defines an interval.

Since every object has a well defined domain, it is possible to combine them in an unambiguous way

.. code:: python

    obs1 = zfit.Space(['x', 'y'])
    obs2 = zfit.Space(['z', 'y'])

    model1 = zfit.pdf.Gauss(obs=obs1, ...)
    model2 = zfit.pdf.Gauss(obs=obs2, ...)

    # creating a composite pdf
    product = model1 * model2
    # OR, equivalently
    product = zfit.pdf.ProductPDF([model1, model2])

The ``product`` is now defined in the space with observables `['x', 'y', 'z']`. Any :py:class:`~zfit.Data` object
to be combined with ``product`` has to be specified in the same space.

.. code:: python

    # create the space
    combined_obs = obs1 * obs2

    data = zfit.Data.from_numpy(obs=combined_obs, ...)

Now we have a :py:class:`~zfit.Data` object that is defined in the same domain as `product` and can be used to build a loss function.

Limits
------

In many places, just defining the observables is not enough and an interval, specified by its limits, is required.
Examples are a normalization range, the limits of an integration or sampling in a certain region.

Simple, 1-dimensional limits can be specified as follows. Operations like addition (creating a space with
two intervals) or combination (increase the dimensionality) are also possible.

.. code:: python

    simple_limit1 = zfit.Space(obs='obs1', limits=(-5, 1))
    simple_limit2 = zfit.Space(obs='obs1', limits=(3, 7.5))

    added_limits = simple_limit1 + simple_limit2

In this case, `added_limits` is now a :py:class:`~zfit.Space` with observable `'obs1'` defined in the intervals
(-5, 1) and (3, 7.5). This can be useful, *e.g.*, when fitting in two regions.
An example of the product of different :py:class:`~zfit.Space` instances has been shown before as ``combined_obs``.


Defining limits
'''''''''''''''

To define simple, 1-dimensional limits, a tuple with two numbers is enough. For anything more complicated,
the definition works as follows:

.. code:: python

    first_limit_lower = (low_1_obs1, low_1_obs2,...)
    first_limit_upper = (up_1_obs1, up_1_obs2,...)

    second_limit_lower = (low_2_obs1, low_2_obs2,...)
    second_limit_upper = (up_2_obs1, up_2_obs2,...)

    ...

    lower = (first_limit_lower, second_limit_lower, ...)
    upper = (first_limit_upper, second_limit_upper, ...)

    limits = (lower, upper)

    space1 = zfit.Space(obs=['obs1', 'obs2', ...], limits=limits)

This defines the area from

* `low_1_obs1` to `up_1_obs1` in the first observable `'obs1'`;
* `low_1_obs2` to `up_1_obs2` in the second observable `'obs2'`;
* ...

the area from

* `low_2_obs1` to `up_2_obs1` in the first observable `'obs1'`;
* `low_2_obs2` to `up_2_obs2` in the second observable `'obs2'`;
* ...

and so on.


A working code example of :py:class:`~zfit.Space` handling is provided in `spaces.py` in
:doc:`examples <../examples/spaces.py>`.
