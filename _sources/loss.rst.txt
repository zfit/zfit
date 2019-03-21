Loss
====

A *loss function* can be defined as a measurement of the discrepancy between the observed data and the predicted data by the fitted function. 
To some extend it can be visualised as a metric of the goodness of a given prediction as you change the settings of your algorithm. 
For example, in a general linear model the loss function is essentially the sum of squared deviations from the fitted line or plane. 
A more useful application in the context of High Energy Physics (HEP) is the Maximum Likelihood Estimator (MLE).
The MLE is a specific type of probability model estimation, where the loss function is the negative log-likelihood (NLL). 

There are two loss functions implementations in the ``zfit`` library, i.e. the :py:class:`~zfit.loss.UnbinnedNLL` and :py:class:`~zit.loss.ExtendedUnbinnedNLL` classes.
These loss functions can then be built using ``pdf.prob``, following a common interface, in which the model, 
the dataset and the fit range (which internally sets ``norm_range`` in the PDF and makes sure data only within that range are used) **must** be given, 
and where parameter constraints in form of a dictionary `{param: constraint}` **may** be given. 
As an example, for unbinned likelihood fits 

.. code-block:: python

    my_loss = zfit.loss.UnbinnedNLL(model,
                                    data,
                                    fit_range=(-10, 10),
                                    constraints={mu: zfit.pdf.Gauss(mu=1., sigma=0.4)})

Additional constraints **may** be passed to the loss object using the ``add_constraint(constraints)`` method.
Moreover, in the event of an extended PDF the loss implementation is identical apart from the use of the ``ExtendedUnbinnedNLL``. 

A very common use case likelihood fits in HEP is the possibility to examine simultaneously different datasets (that can be independent or somehow correlated). 
To build loss functions for simultaneous fits, the addition operation, 
either through the `my_loss.add` method or through the `+` operator, can be used (the particular combination that is performed depends on the type of loss function). 
The same result can be achieved by passing a list of PDFs on instantiation, along with the same number of datasets and fit ranges.
As an example, let's consider a typical case  

.. code-block:: python
 
    # Adding a list of models, data and observable ranges
    my_loss_sim = zfit.loss.UnbinnedNLL(model=[models], data=[datasets], fit_range=[obsRange])

    # OR summing two loss functions
    my_loss_sim_operator = my_loss1 + my_loss2

