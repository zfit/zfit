.. _loss-section:

====
Loss
====

A *loss function* can be defined as a measurement of the discrepancy between the observed data and the predicted data by the fitted function.
To some extent it can be visualised as a metric of the goodness of a given prediction as you change the settings of your algorithm.
For example, in a general linear model the loss function is essentially the sum of squared deviations from the fitted line or plane.
A more useful application in the context of High Energy Physics (HEP) is the Maximum Likelihood Estimator (MLE).
The MLE is a specific type of probability model estimation, where the loss function is the negative log-likelihood (NLL).

In zfit, loss functions inherit from the :py:class:`~zfit.loss.BaseLoss` class and they follow a common interface, in which the model,
the dataset **must** be given, and
where parameter constraints in form of a dictionary ``{param: constraint}`` **may** be given.
As an example, we can create an unbinned negative log-likelihood loss (:py:class:`~zfit.loss.UnbinnedNLL`) from the model described in the :ref:`Basic model section <basic-model>` and the data from the :ref:`Data section <data-section>`:

.. code-block::

    my_loss = zfit.loss.UnbinnedNLL(model_cb, data)

Adding constraints
------------------

Constraints (or, in general, penalty terms) can be added to the loss function either by using the ``constraints`` keyword when creating the loss object or by using the :py:func:`~zfit.loss.BaseLoss.add_constraints` method.
These constraints are specified as a list of penalty terms, which can be any object inheriting from :py:class:`~zfit.constraint.BaseConstraint` that is simply added to the calculation of the loss.

Useful implementations of penalties can be found in the :py:mod:`zfit.constraint` module.
For example, if we wanted to add a gaussian constraint on the ``mu`` parameter of the previous model, we would write:

.. code-block:: pycon

    >>> constraint = zfit.constraint.GaussianConstraint(params=mu,
    >>>                                                observation=5279.,
    >>>                                                uncertainty=10.)

    >>> my_loss = zfit.loss.UnbinnedNLL(model_cb,
    >>>                                 data,
    >>>                                 constraints=constraint)

Custom penalties can also be added to the loss function, for instance if you want to set limits on a parameter:

.. code-block:: pycon

    >>> def custom_constraint(param):
            max_value = 5400
            return tf.cond(tf.greater_equal(param, max_value), lambda: 10000., lambda: 0.)

The custom penalty needs to be a ``SimpleConstraint`` to be added to the loss function whereas ``mu`` will be used
as the argument to the constraint

.. code-block:: pycon

    >>> simple_constraint = zfit.constraint.SimpleConstraint(custom_constraint, params=mu)
    >>> my_loss.add_constraints(simple_constraint)

In this example if the value of ``param`` is larger than ``max_value`` a large value is added the loss function
driving it away from the minimum.



Simultaneous fits
-----------------

There are currently two loss function implementations in the ``zfit`` library, the :py:class:`~zfit.loss.UnbinnedNLL` and :py:class:`~zfit.loss.ExtendedUnbinnedNLL` classes, which cover non-extended and extended negative log-likelihoods.

A very common use case of likelihood fits in HEP is the possibility to examine simultaneously different datasets (that can be independent or somehow correlated).
To build loss functions for simultaneous fits, the addition operator can be used (the particular combination that is performed depends on the type of loss function):

.. code-block:: pycon

   >>> models = [model1, model2]
   >>> datasets = [data1, data2]
   >>> my_loss1 = zfit.loss.UnbinnedNLL(models[0], datasets[0], fit_range=(-10, 10))
   >>> my_loss2 = zfit.loss.UnbinnedNLL(models[1], datasets[1], fit_range=(-10, 10))
   >>> my_loss_sim_operator = my_loss1 + my_loss2

The same result can be achieved by passing a list of PDFs on instantiation, along with the same number of datasets:

.. code-block:: pycon

   >>> # Adding a list of models and datasets
   >>> my_loss_sim = zfit.loss.UnbinnedNLL(model=[model1, model2, ...], data=[data1, data2, ...])
