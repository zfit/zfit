Minimization
============

Minimizer objects are the last key element in the API framework of ``zfit``. 
In particular, these are connected to the loss function and have an internal state that can be queried at any moment.

The ``zfit`` library is designed such that it is trivial to introduce new sets of minimizers.
The only requirement in its initialisation is that a loss function **must** be given. 
Additionally, the parameters to be minimize, the tolerance, its name, as well as any other argument needed to configure the particular algorithm **may** be given. 

Baseline minimizers
'''''''''''''''''''

There are three minimizers currently maintained in the package: ``Minuit``, ``Scipy`` and ``Adam`` TensorFlow optimiser. 
Let's show how these can be initialised:

.. code-block:: python

    # Minuit minimizer
    minimizer_minuit = zfit.minimize.MinuitMinimizer()

    # Scipy minimizer
    minimizer_scipy = zfit.minimize.ScipyMinimizer()

    # Adam's Tensorflow minimizer
    minimizer_adam = zfit.minimize.AdamMinimizer()

We also have available a wrapper for TensorFlow optimisers, such that new ideas are easily integrate in the framework. 
For instance, the Adam minimizer could have been initialised by 

.. code-block:: python

    # Adam's TensorFlor optimiser using a wrapper
    minimizer_wrapper = zfit.minimize.WrapOptimizer(tf.train.AdamOptimizer())

Finally, any of these minimizers can then be used to minimize the loss function, e.g.

.. code-block:: python

    # E.g. in the case of Minuit
    result = minimizer_minuit.minimize(loss=nll) 

The choice of which parameters of your model should be floating in the fit can also be determined at this stage

.. code-block:: python

    # In the case of a Gaussian (e.g.)
    result = minimizer_minuit.minimize(loss=nll, params=[mu, sigma]) 

**Only** the parameters provided are floated in the optimisation process. 
If this argument is not provided or ``params=None``, all the parameters in the fit are floated in the minimization process. 

Minimizers information
''''''''''''''''''''''

The internal state of the Minimizer is stored in a MinimizerState object, 
which provides access to the Estimated Distance to the Minimum (EDM), 
the value at the minimum and its status through the EMD, fmin and status properties, respectively. 
For instance

.. code-block:: python

    # Function minimum
    result.fmin

    # Whether converged or not
    result.converged

    # Full minimizer information
    result.info

Additionally, the parameters of the minimizer can be accessed through the ``result.params``, 
which provides all the necessary information on them. 


