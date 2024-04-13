============================
Space, Observable and Range
============================

Inside zfit, :py:class:`~zfit.Space` defines the domain of objects by specifying the observables/axes and *maybe* also
the limits. Any model and data needs to be specified in a certain domain, which is usually done using the
``obs`` argument. It is crucial that the axis used by the observable of the data and the model match, and this matching is
handle by the :py:class:`~zfit.Space` class.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import zfit
    from zfit import z
    import numpy as np


.. code-block::

    obs = zfit.Space("x")
    model = zfit.pdf.Gauss(obs=obs, ...)
    data = zfit.Data.from_numpy(obs=obs, ...)

Definitions
-----------
**Space**: an *n*-dimensional definition of a domain (either by using one or more observables or axes),
with or without limits.

.. note:: Difference to RooFit
    :class: dropdown

    *compared to ``RooFit``, a space is **not** the equivalent of an observable but rather corresponds
    to an object combining **a set** of observables (which of course can be of size 1). Furthermore,
    there is a **strong** distinction in zfit between a :py:class:`~zfit.Space` (or observables)
    and a :py:class:`~zfit.Parameter`, both conceptually and in terms of implementation and usage.*

**Observable**: a string defining the axes; a named axes.

*(for advanced usage only, can be skipped on first read)*
**Axis**: integer defining the axes *internally* of a model. There is always a mapping of observables <-> axes *once inside a model*.

**Limit** The range on a certain axis. Typically defines an interval. In fact, there are two times of limits:
 * **rectangular**: This type is the usual limit as e.g. ``(-2, 5)`` for a simple, 1 dimensional interval.

A space should usually be created with limits that define the default space of an object.
This correspond for example to the default normalization range ``norm`` or sampling range.

.. code-block::


    obs1 = zfit.Space("x", 0, 1)
    obs2 = zfit.Space("y", -4, 1)


    model1 = zfit.pdf.Gauss(obs=obs1, ...)
    model2 = zfit.pdf.Gauss(obs=obs2, ...)

    # creating a composite pdf
    product = model1 * model2
    # OR, equivalently
    product = zfit.pdf.ProductPDF([model1, model2])

    assert obs1 * obs2 == product.space

The ``product`` is now defined in the space with observables ``['x', 'y']``. Any :py:class:`~zfit.Data` object
to be combined with ``product`` has to be specified in the same space.

.. code-block::

    # create the space
    combined_obs = obs1 * obs2

    data = zfit.Data.from_numpy(obs=combined_obs, ...)

Now we have a :py:class:`~zfit.Data` object that is defined in the same domain as ``product``
and can be used to build a loss function.

Limits
------

In many places, just defining the observables is not enough and an interval, specified by its limits, is required.
Examples are a normalization range, the limits of an integration or sampling in a certain region.



Using the limits
'''''''''''''''''

To use the limits of any object, the methods :py:meth:`~zfit.Space.inside`
(to test if values are inside or outside of the boundaries)
and :py:meth:`~zfit.Space.filter` can be used.
