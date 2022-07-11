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
 * **rectangular**: This type is the usual limit as e.g. ``(-2, 5)`` for a simple, 1 dimensional interval. It is
   rectangular. This can either be given as ``limits`` of a :py:class:`~zfit.Space` or as ``rect_limits``.
 * **functional**: In order to define arbitrary limits, a function can be used that receives a tensor-like
   object ``x`` and returns ``True`` on every position that is inside the limits, ``False`` for every value outside.
   When a functional limit is given, rectangular limits that contain the functional limit as a subset **must** be
   defined as well.


Since every object has a well defined domain, it is possible to combine them in an unambiguous way.
While not enforced, a space should usually be created with limits that define the default space of an object.
This correspond for example to the default normalization range ``norm_range`` or sampling range.

.. code-block::

    lower1, upper1 = [0, 1], [2, 3]
    lower2, upper2 = [-4, 1], [10, 3]
    obs1 = zfit.Space(['x', 'y'], limits=(lower1, upper2))
    obs2 = zfit.Space(['z', 'y'], limits=(lower2, upper2))

    model1 = zfit.pdf.Gauss(obs=obs1, ...)
    model2 = zfit.pdf.Gauss(obs=obs2, ...)

    # creating a composite pdf
    product = model1 * model2
    # OR, equivalently
    product = zfit.pdf.ProductPDF([model1, model2])

    assert obs1 * obs2 == product.space

The ``product`` is now defined in the space with observables ``['x', 'y', 'z']``. Any :py:class:`~zfit.Data` object
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

Simple, 1-dimensional limits can be specified as follows. Operations like addition (creating a space with
two intervals) or combination (increase the dimensionality) are also possible.

.. jupyter-execute::

    simple_limit1 = zfit.Space(obs='obs1', limits=(-5, 1))
    simple_limit2 = zfit.Space(obs='obs1', limits=(3, 7.5))

    added_limits = simple_limit1 + simple_limit2

In this case, ``added_limits`` is now a :py:class:`zfit.Space` with observable ``obs1'`` defined in the intervals
(-5, 1) and (3, 7.5). This can be useful, *e.g.*, when fitting in two regions.
An example of the product of different :py:class:`zfit.Space` instances has been shown before as ``combined_obs``.

Functional limits
'''''''''''''''''

Limits can be defined by a function that returns whether a value is inside the boundaries or not **and** rectangular
limits (note that specifying ``rect_limit`` does *not* enforce them, the function itself has to take care of that).

This example specifies the bounds between (-4, 0.5) with the ``limit_fn`` (which, in this simple case, could be better
achieved by directly specifying them as rectangular limits).

.. code:: python

    def limit_fn(x):
        x = z.unstack_x(x)
        inside_lower = tf.greater_equal(x, -4)
        inside_upper = tf.less_equal(x, 0.5)
        inside = tf.logical_and(inside_lower, inside_upper)
        return inside

    space = zfit.Space(obs='obs1', limits=limit_fn, rect_limits=(-5, 1))


Combining limits
''''''''''''''''

To define simple, 1-dimensional limits, a tuple with two numbers or a functional
limit in 1 dimension is enough. For anything more complicated,
the operators product ``*`` or addition ``+`` respectively their functional API
:py:func:`zfit.dimension.combine_spaces`
and :py:func:`zfit.dimension.add_spaces` can be used.


A working code example of :py:class:`~zfit.Space` handling is provided in :ref:`spaces.py <spaces-example>`.



Using the limits
'''''''''''''''''

To use the limits of any object, the methods :py:meth:`~zfit.Space.inside`
(to test if values are inside or outside of the boundaries)
and :py:meth:`~zfit.Space.filter` can be used.

The rectangular limits can also direclty be accessed by ``rect_limits``, ``rect_lower`` or ``rect_upper``.
The returned shape is of
``(n_events, n_obs)``, for the lower respectively upper limit (``rect_limits`` is a tuple of ``(lower, upper)``).
This should be used with caution and only if the rectangular limits are desired.
