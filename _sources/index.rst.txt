================================
zfit: scalable pythonic fitting
================================

The zfit package is a model fitting library based on `TensorFlow <https://www.tensorflow.org/>`_ and optimised for simple and direct manipulation of probability density functions. The main focus is on the scalability, parallelisation and a user friendly experience framework (no cython, no C++ needed to extend). The basic idea is to offer a pythonic oriented alternative to 
the very successful RooFit library from the `ROOT <https://root.cern.ch/>`_ data analysis package. While RooFit has provided a stable platform for most of the needs of the High Energy Physics (HEP) community in the last few years, it has become increasingly difficult to integrate all the developments in the scientific Python ecosystem into RooFit due to its monolithic nature. Conversely, the core of zfit aims at becoming a solid ground for model fitting while providing enough flexibility to incorporate state-of-art tools and to allow scalability going to larger datasets.
This challenging task is tackled by following two basic design pillars:   
  
- The skeleton and extension of the code is minimalist, simple and finite:
  the zfit library is exclusively designed for the purpose of model fitting and sampling---opposite to the self-contained RooFit/ROOT frameworks---with no attempt to extend its functionalities to features such as statistical methods or plotting.
  This design philosophy is well exemplified by examining maximum likelihood fits: while zfit works as a backend for likelihood fits and can be integrated to packages such as `lauztat <https://github.com/marinang/lauztat>`_ and `matplotlib <https://matplotlib.org/>`_, RooFit performs the fit, the statistical treatment and plotting within.
  This wider scope of RooFit results in a lack of flexibility with respect to new minimisers, statistic methods and, broadly speaking, any new tool that might come.

- Another paramount aspect of zfit is its design for optimal parallelisation and scalability. Even though the choice of TensorFlow as backend introduces a strong software dependency, its use provides several interesting features in the context of model fitting.
  The key concept is that TensorFlow is built under the `dataflow programming model <https://en.wikipedia.org/wiki/Dataflow_programming>`_.
  Put it simply, TensorFlow creates a computational graph with the operations as the nodes of the graph and tensors to its edges. Hence, the computation only happens when the graph is executed in a session, which simplifies the parallelisation by identifying the dependencies between the edges and operations or even the partition across multiple devices (more details can be found in the `TensorFlow guide <https://www.tensorflow.org/guide/>`_).
  The architecture of zfit is built upon this idea and it aims to provide a high level interface to these features, *i.e.*, most of the operations of graphs and evaluations are hidden for the user, leaving a natural and friendly model fitting and sampling experience. 

The zfit package is Free software, using an Open Source license. Both the software and this document are works in progress.
Source code can be found in `our github page <https://github.com/zfit/zfit/>`_.

.. toctree::
    :maxdepth: 2

    getting_started
    downloading
    contributing
    space
    parameter
    model
    data
    loss
    minimize

.. toctree::
    :maxdepth: 1

    API



