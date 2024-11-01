Strategy
#############

A strategy helps a minimizer in some situations - such as when encountering NaNs - and hooks
into the loss evaluation. They can be used to create a
:class:`~zfit.minimize.Minimizer` with a specific strategy.

.. autosummary::
    :toctree: _generated/strategy

    zfit.minimize.PushbackStrategy
    zfit.minimize.DefaultStrategy
    zfit.minimize.DefaultToyStrategy
