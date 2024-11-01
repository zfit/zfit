HS3 serialization
-----------------

Some objects in zfit can be serialized using a `HS3-like <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_ serialization. This is a human-readable, human-editable, and language-independent serialization format that can be used to store and load objects.

Not all objects can be serialized.

.. autosummary::
    :toctree: _generated/hs3

    zfit.hs3.dumps
    zfit.hs3.loads
    zfit.hs3.dump
    zfit.hs3.load
