dill
--------------------------------------------------------------------------------

Drop-in replacement for `dill <https://github.com/uqfoundation/dill#dill>`_, a more powerful version of pickle: Contains a thin wrapper around dumping objects with dill.

Simply use ``import zfit.dill as dill`` instead of ``import dill`` to use the zfit version of dill or use the functions directly with ``zfit.dill.dump`` and ``zfit.dill.dumps`` (and others).

In general, ``dill.dump`` and ``dill.dumps`` work directly with zfit objects. However, due to some issue with the
garbage collection, they can fail unpredictably. This module, mostly the functions ``zfit.dill.dump`` and ``zfit.dill.dumps`` are thin wrappers around dill that circumvent this issue if it arises.


The module contains *all* functions and classes of dill. Here, only the altered functions are documented.

.. autosummary::
    :toctree: _generated/dill

    zfit.dill.dump
    zfit.dill.dumps
