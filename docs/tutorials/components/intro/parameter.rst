Parameter
=========

Several objects in zfit, most importantly models, have one or more parameter which typically
parametrise a function or distribution. There are two different kinds of parameters in zfit:

* Independent: can be changed in a fit (or explicitly be set to ``fixed``).
* Dependent: **cannot** be directly changed but *may* depend on independent parameters.

Unique names
-------------

Parameters in zfit are global, unique objects. No Parameters with the same name can therefore exist as its meaning would
be ambiguous. If a new parameter with the same name will be created, a :class:`~zfit.exception.NameAlreadyTakenError`
will be raised.

For Jupyter notebooks, see also :ref:`about parameters in Jupyter <params-in-jupyter>` for
additional information

Independent Parameter
---------------------

To create a parameter that can be changed, *e.g.*, to fit a model, a :py:class:`~zfit.Parameter` has to
be instantiated.

The syntax is as follows:

.. code:: python

    param1 = zfit.Parameter("unique_param_name", start_value[, lower_limit, upper_limit])

Furthermore, a ``step_size`` can be specified. If not, it is set to a default value around 0.1.
:py:class:`~zfit.Parameter` can have limits (tested with :py:meth:`~zfit.Parameter.has_limits`), which will
clip the value to the limits given by :py:meth:`~zfit.Parameter.lower_limit` and
:py:meth:`~zfit.Parameter.upper_limit`.

.. note:: Comparison to RooFit

    While this closely follows the RooFit syntax, it is very important to note that the optional limits
    of the parameter behave differently:
    if not given, the parameter will be "unbounded", not fixed (as in RooFit).
    Parameters are therefore floating by default, but their value can be fixed by setting the attribute
    ``floating`` to ``False`` or already specifying it in the init.

The value of the parameter can be changed with the :py:func:`~zfit.Parameter.set_value` method.
Using this method as a context manager, the value can also temporarily changed.
However, be aware that anything _dependent_ on the parameter will have a value with the
parameter evaluated with the new value at run-time:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import zfit

.. jupyter-execute::

    mu = zfit.Parameter("mu_one", 1)  # no limits, but FLOATING (!)
    with mu.set_value(3):
        print(f'before {mu}')

    # here, mu is again 1
    print(f'after {mu}')


Dependent Parameter
-------------------

A parameter can be composed of several other parameters. They can be used equivalently to :py:class:`~zfit.Parameter`.

.. jupyter-execute::

    mu2 = zfit.Parameter("mu_two", 7)

    def dependent_func(mu, mu2):
        return mu * 5 + mu2  # or any kind of computation
    dep_param = zfit.ComposedParameter("dependent_param", dependent_func, params=[mu, mu2])

    print(dep_param.get_params())


A special case of the above is :py:class:`~zfit.ComplexParameter`: it
provides a few special methods (like :py:func:`~zfit.ComplexParameter.real`,
:py:func:`~zfit.ComplexParameter.conj` etc.)
to easier deal with complex numbers.
Additionally, the :py:func:`~zfit.ComplexParameter.from_cartesian` and :py:func:`~zfit.ComplexParameter.from_polar`
methods can be used to initialize polar parameters from floats, avoiding the need of creating complex
:py:class:`tf.Tensor` objects.


.. _params-in-jupyter:

Parameters in Jupyter
----------------------

Parameters are unique, global objects. This can conflict with the typical workflow in a jupyter notebook as cells are
often executed multiple times. If a cell that creates a parameter is executed again (meaning a parameter with the same
name as already existing should be created), it raises a :class:`~zfit.exception.NameAlreadyTakenError`
(there is `an extensive discussion of the why <https://github.com/zfit/zfit/issues/186>`_)

To circumvent this, which comes from the fact that Jupyter is stateful, there are a few ways:

- if possible, simply rerun everything.
- move the creation of the variables into a separate cell at the beginning. Remember that you can set a value on a
  variable anytime using :meth:`~zfit.Parameter.set_value` which can be called as often as desired.
- create a wrapper that returns the same parameter again if it exists. With this way it is clear what is done
  and it is convenient to use as a de-facto drop-in replacement for :class:~`zfit.Parameter` (using it in
  other places except for exploratory work may has unintended side-consequences)

Example wrapper:

.. jupyter-execute::

    all_params = {}

    def get_param(name, value=None, lower=None, upper=None, step_size=None, **kwargs):
        """Either create a parameter or return existing if a parameter with this name already exists.

        If anything else than *name* is given, this will be used to change the existing parameter.

        Args:
            name: Name of the Parameter
            value : starting value
            lower : lower limit
            upper : upper limit
            step_size : step size

        Returns:
            ``zfit.Parameter``
        """
        if name in all_params:
            parameter = all_params[name]
            if lower is not None:
                parameter.lower = lower
            if upper is not None:
                parameter.upper = upper
            if step_size is not None:
                parameter.step_size = step_size
            if value is not None:
                parameter.set_value(value)
            return parameter

        # otherwise create new one
        parameter = zfit.Parameter(name, value, lower, upper, step_size)
        all_params[name] = parameter
        return parameter

This wrapper can then be used instead of :class:`~zfit.Parameter` as

.. jupyter-execute::

    param1 = get_param('param1', 5., 3., 10., step_size=5)
    # Do something with it
    param2 = get_param('param1', 3., 1., 10.)  # will have step_size 5 as we don't change that
    assert param2 is param1
