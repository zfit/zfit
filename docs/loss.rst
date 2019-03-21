.. _loss:

====
Loss
====

A *loss function* can be defined as a measurement of the discrepancy between the observed data and the predicted data by the fitted function. 
To some extent it can be visualised as a metric of the goodness of a given prediction as you change the settings of your algorithm. 
For example, in a general linear model the loss function is essentially the sum of squared deviations from the fitted line or plane. 
A more useful application in the context of High Energy Physics (HEP) is the Maximum Likelihood Estimator (MLE).
The MLE is a specific type of probability model estimation, where the loss function is the negative log-likelihood (NLL). 

In zfit, loss functions inherit from the :py:class:`~zfit.core.loss.BaseLoss` class and they follow a common interface, in which the model, 
the dataset and the fit range (which internally sets ``norm_range`` in the PDF and makes sure data only within that range are used) **must** be given, and 
where parameter constraints in form of a dictionary `{param: constraint}` **may** be given.
As an example, we can create an unbinned negative log-likelihood loss (:py:class:`~zfit.core.loss.UnbinnedNLL`) from the model described in the :ref:`Basic model section <basic-model>` and the data from the :ref:`Data section <data-section>`:

.. code-block:: pycon

    >>> my_loss = zfit.loss.UnbinnedNLL(model_cb,
    >>>                                 data,
    >>>                                 fit_range=(-10, 10))

Adding constraints
------------------

Constraints (or, in general, penalty terms) can be added to the loss function either by using the ``constraints`` keyword when creating the loss object or by using the :py:func:`~zfit.core.loss.BaseLoss.add_constraints` method.
These constraints are specified as a list of penalty terms, which can be any ``tf.Tensor`` object that is simply added to the calculation of the loss.

Useful implementations of penalties can be found in the :py:mod:`zfit.constraint` module.
For example, if we wanted to add adding a gaussian constraint on the ``mu`` parameter of the previous model, we would write: 

.. code-block:: pycon

    >>> my_loss = zfit.loss.UnbinnedNLL(model_cb,
    >>>                                 data,
    >>>                                 fit_range=(-10, 10),
    >>>                                 constraints=zfit.constraint.nll_gaussian(params=mu,
    >>>                                                                          mu=5279.,
    >>>                                                                          sigma=10.))


Simultaneous fits
-----------------

There are currently two loss functions implementations in the ``zfit`` library, the :py:class:`~zfit.core.loss.UnbinnedNLL` and :py:class:`~zfit.core.loss.ExtendedUnbinnedNLL` classes, which cover non-extended and extended negative log-likelihoods.

A very common use case likelihood fits in HEP is the possibility to examine simultaneously different datasets (that can be independent or somehow correlated). 
To build loss functions for simultaneous fits, the addition operator can be used (the particular combination that is performed depends on the type of loss function):

.. code-block:: pycon
 
   >>> models = [model1, model2]
   >>> datasets = [data1, data2]
   >>> my_loss1 = zfit.loss.UnbinnedNLL(models[0], datasets[0], fit_range=(-10, 10))
   >>> my_loss2 = zfit.loss.UnbinnedNLL(models[1], datasets[1], fit_range=(-10, 10))
   >>> my_loss_sim_operator = my_loss1 + my_loss2

The same result can be achieved by passing a list of PDFs on instantiation, along with the same number of datasets and fit ranges:

.. code-block:: pycon
 
   >>> # Adding a list of models, data and observable ranges
   >>> my_loss_sim = zfit.loss.UnbinnedNLL(model=[models], data=[datasets], fit_range=[obsRange])

