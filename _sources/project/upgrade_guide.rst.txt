================================
Upgrade guide
================================


Upgrade from zfit 0.3.x to 0.4.0
================================

zfit moved from TensorFlow 1.x to 2.x. The main difference is that in 1.x, you would mostly built
a graph all the time and execute it when needed. In TF 2.x, this has gone and happens implicitly
if a function is decorated with the right decorator. But it is also possible to build no graph at all
and execute the code _eagerly_, just as Numpy would. So writing just TF 2.x code is "no different", if not wrapped
by a :py:func:`tf.function`, than executing Numpy code.

In short: write TF 2.x as if you would write Numpy. If something is supposed to _change_, it has to be
newly generated each time, e.g. be a function that can be called.

zfit offers objects that still keep track of everything.

Consequences for zfit:

Dependents
----------

this implies that zfit does not rely on the graph structure anymore.
Therefore, dependencies have to be given manually (although in the future, certain automatitions
can surely be added).

Affected from this is the :py:class:`~zfit.ComposedParameter`. Instead of giving a Tensor,
a function returning a value has to be given _and_ the dependents have to be specified
explicitly.

.. code-block:: python

    mu = zfit.Parameter(...)
    shift = zfit.Parameter(...)
    def shifted_mu_func():
        return mu + shift

    shifted_mu = zfit.params.ComposedParameter(shifted_mu_func, dependents=[mu, shift])



The same is true for the :py:class:`~zfit.loss.SimpleLoss`
