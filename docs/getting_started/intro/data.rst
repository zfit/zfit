.. _data-section:

====
Data
====

An easy and fast data manipulation are among the crucial aspects in High Energy Particle physics data analysis.
With the increasing data availability (e.g. with the advent of LHC), this challenge has been pursued in different
manners. Common strategies vary from multidimensional arrays with attached row/column labels (e.g. ``DataFrame`` in *pandas*) or compressed binary formats (e.g. ROOT). While each of these data structure designs have their own advantages in terms of speed and acessibility, the data concept inplemented in ``zfit`` follows closely the features of ``DataFrame`` in *pandas*.

The :py:class:`~zfit.Data` class provides a simple and structured access/manipulation of *data* -- similarly to concept of multidimensional arrays approach from *pandas*. The key feature of :py:class:`~zfit.Data` is its relation to the :py:class:`~zfit.Space` or more explicitly its axis or name. A more equally convention is to name the role of the :py:class:`~zfit.Space` in this context as the *observable* under investigation. Note that no explicit range for the :py:class:`~zfit.Space` is required at the moment of the data definition, since this is only required at the moment some calculation is needed (e.g. integrals, fits, etc).

Import dataset from a ROOT file
--------------------------------

With the proliferation of the ROOT framework in the context of particle physics, it is often the case that the user will have access to a ROOT file in their analysis. A simple method has been used to handle this conversion:

.. code-block::

    data = zfit.Data.from_root(root_file, root_tree, branches)

where ``root_file`` is the path to the ROOT file, ``root_tree`` is the tree name and branches are the list (or a single) of branches that the user wants to import from the ROOT file.

From the default conversion of the dataset there are two optional funcionalities for the user, i.e. the use of weights and the rename of the specified branches. The nominal structure follows:

.. code-block:: pycon

    >>> data = zfit.Data.from_root(root_file,
    ...                                root_tree,
    ...                                branches,
    ...                                branches_alias=None,
    ...                                weights=None)

The ``branches_alias`` can be seen as a list of strings that renames the original ``branches``. The ``weights`` has two different implementations: (1) either a 1-D column is provided with shape equals to the data (nevents) or (2) a column of the ROOT file by using a string corresponding to a column. Note that in case of multiple weights are required, the weight manipulation has to be performed by the user beforehand, e.g. using Numpy/pandas or similar.

.. note::

    The implementation of the ``from_root`` method makes uses of the uproot packages,
    which uses Numpy to cast blocks of data from the ROOT file as Numpy arrays in time optimised manner.
    This also means that the *goodies* from uproot can also be used by specifying the root_dir_options,
    such as cuts in the dataset. However, this can be applied later when examining the produced dataset
    and it is the advised implementation of this.

Import dataset from a pandas DataFrame or Numpy ndarray
-------------------------------------------------------

A very simple manipulation of the dataset is provided via the pandas DataFrame. Naturally this is simplified since the :py:class:`~zfit.Space` (observable) is not mandatory, and can be obtained directly from the columns:

.. code-block:: pycon

    >>> data = zfit.Data.from_pandas(pandas.DataFrame,
    ...                              obs=None,
    ...                              weights=None)

In the case of Numpy, the only difference is that as input is required a numpy ndarray and the :py:class:`~zfit.Space` (obs) is mandatory:

.. code-block:: pycon

    >>> data = zfit.Data.from_numpy(numpy.ndarray,
    ...                             obs,
    ...                             weights=None)
