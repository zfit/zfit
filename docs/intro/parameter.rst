Parameter
=========

Several objects in zfit, most importantly models, have one or more parameter which typically
parametrise a function or distribution. There are two different kinds of parameters in zfit:

* Independent: can be changed in a fit (or explicitly be set to `fixed`).
* Dependent: **cannot** be directly changed but _may_ depend on independent parameters.


Independent Parameter
---------------------

To create a parameter that can be changed, *e.g.*, to fit a model, a :py:class:`~zfit.Parameter` has to
be instantiated.

The syntax is as follows:

.. code:: python

    param1 = zfit.Parameter("param_name_human_readable", start_value[, lower_limit, upper_limit])

Furthermore, a ``step_size`` can be specified. If not, it is set to a default value around 0.001.
:py:class:`~zfit.Parameter` can have limits (tested with :py:meth:`~zfit.Parameter.has_limits`), which will
clip the value to the limits given by :py:meth:`~zfit.Parameter.lower_limit` and
:py:meth:`~zfit.Parameter.upper_limit`.
While this closely follows the RooFit syntax, it is very important to note that the optional limits
of the parameter behave differently:
if not given, the parameter will be "unbounded", not fixed (as in RooFit).
Parameters are therefore floating by default, but their value can be fixed by setting the attribute
``floating`` to ``False`` or already specifying it in the init.

The value of the parameter can be changed with the :py:func:`~zfit.Parameter.set_value` method.
Using this method as a context manager, the value can also temporarily changed.
However, be aware that anything _dependent_ on the parameter will have a value with the
parameter evaluated with the new value at run-time:

.. code:: pycon

    >>> mu = zfit.Parameter("mu_one", 1)  # no limits, but FLOATING (!)
    >>> with mu.set_value(3):
    ...    # in here, mu has the value 3
    ...    mu_val = zfit.run(mu)  # 3
    ...    five_mu = 5 * mu
    ...    five_mu_val = zfit.run(five_mu)  # is evaluated with mu = 5. -> five_mu_val is 15

    >>> # here, mu is again 1
    >>> mu_val_after = zfit.run(mu)  # 1
    >>> five_mu_val_after = zfit.run(five_mu)  # is evaluated with mu = 1! -> five_mu_val_after is 5


Dependent Parameter
-------------------

A parameter can be composed of several other parameters. We can use any :py:class:`~tf.Tensor` for that
and the dependency will be detected automatically. They can be used equivalently to :py:class:`~zfit.Parameter`.

.. code:: pycon

    >>> mu2 = zfit.Parameter("mu_two", 7)
    >>> dependent_tensor = mu * 5 + mu2  # or any kind of computation
    >>> dep_param = zfit.ComposedParameter("dependent_param", dependent_tensor)

    >>> dependents = dep_param.get_dependents()  # returns set(mu, mu2)


A special case of the above is :py:class:`~zfit.ComplexParameter`: it takes a complex :py:class:`tf.Tensor` as input and provides a few special methods (like :py:func:`~zfit.ComplexParameter.real`, :py:func:`~zfit.ComplexParameterconj` etc.) to easier deal with them.
Additionally, the :py:func:`~zfit.ComplexParameter.from_cartesian` and :py:func:`~zfit.ComplexParameter.from_polar` methods can be used to initialize polar parameters from floats, avoiding the need of creating complex :py:class:`tf.Tensor` objects.
