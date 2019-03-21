Parameter
=========

Several objects in ``zfit``, most importantly models, have one or more parameter which typically
parametrize a function or distribution. There are two different kind of parameters in ``zfit``:
* independent: can be changed in a fit (or explicitly be set to `fixed`)
* dependent: can **not** be directly changed but _may_ depends on independent parameters


Independent Parameter
=====================

To create a parameter that can be changed, e.g. to fit a model, a :py:class:`~zfit.Parameter` has to
be instantiated.

The syntax is as follows (and is the same as in `RooFit`)
.. code:: python

    param1 = zfit.Parameter("param_name_human_readable", start_value[, lower_limit, upper_limit])

The value of the parameter can be changes by `set_value`. Using a context manager, the value is
temporarily changed. Be aware that anything _dependent_ on the parameter will have a value with the
parameter evaluated with the new value an run-time.

.. code:: python

    mu = zfit.Parameter("mu_one", 1)  # no limits
    with mu.set_value(3):
        # in here, mu has the value 3
        mu_val = zfit.run(mu)  # 3
        five_mu = 5 * mu
        five_mu_val = zfit.run(five_mu)  # is evaluated with mu = 5. -> five_mu_val is 15

    # here, mu is again 1
    mu_val_after = zfit.run(mu)  # 1
    five_mu_val_after = zfit.run(five_mu)  # is evaluated with mu = 1! -> five_mu_val_after is 5

While a dependent parameter is usually free, it can be fixed by setting the attribute `floating` to `False`.

Limits
''''''

:py:class:`~zfit.Parameter` can have limits (test with :py:method:`~zfit.Parameter.has_limits) that will
clip the value inside the limits given by `:py:method:`~zfit.Parameter.lower_limit` and
`:py:method:`~zfit.Parameter.upper_limit`.


Dependent Parameter
-------------------

A parameter can be composed of several other parameters. We can use any :py:class:`~tf.Tensor` for that
and the dependency will be detected automatically. They can be used equivalently to :py:class:`~zfit.Parameter`.

.. code:: python

    mu2 = zfit.Parameter("mu_two", 7)
    dependent_tensor = mu * 5 + mu2  # or any kind of computation
    dep_param = zfit.ComposedParameter("dependent_param", dependent_tensor)

    dependents = dep_param.get_dependents()  # returns set(mu, mu2)


A somewhat special case of the above are :py:class:`~zfit.ComplexParameter`. It takes a complex
:py:class:`~tf.Tensor` as an input and has a few special methods (like `real`, `conj` etc.) to
easier deal with complex parameters.
