Data
----

Data can be unbinned or binned.  If unbinned, the data is stored in a ``Data`` object while binned data,
or histogram-like data, is stored in a ``BinnedData`` object.

The binned data is built closely to the histograms in the
`boost-histogram <https://boost-histogram.readthedocs.io/en/latest/>`_ and especially
`Hist <https://github.com/scikit-hep/hist>`_ libraries and has to and from methods to seamlessly go back and
forth between the libraries. Furthermore, ``BinnedData`` implements the
`Unified Histogram Interface, UHI <https://github.com/scikit-hep/uhi>`_ and zfit often expects only an
object that implements the UHI.

.. autosummary::
    :toctree: _generated/data

    zfit.data.Data
    zfit.data.BinnedData
    zfit.data.RegularBinning
    zfit.data.VariableBinning
