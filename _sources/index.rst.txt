Welcome to zfit's documentation!
======================================

The ``zfit`` package is a model fitting library based on ``TensorFlow`` and optimised for simple and direct manipulation of probability density functions. The main focus is on the scalability, parallelisation and a user friendly experience framework (no cython, no C++ needed to extend). The basic idea is to provide a pythonic oriented alternative to 
the very successful ``RooFit`` library from the ROOT (or even pyROOT) data analysis package. While ``RooFit`` has provided a stable platform for most of the needs of the High Energy Physics (HEP) community in the last few years, there are several developments in the machine learning (ML) community that have not been followed by this library. In this sense, the core of ``zfit`` is designed to provide this interface between the solid structure of ``RooFit`` with the state-of-art tools of ML. This challenge task can only be achieved with two paramount ansatz:   
  
  * The skeleton and extension of the code is minimalist/simple and finite. This is conceptually different from ``ROOT/RooFit`` philosophies, *i.e.* the ``zfit`` library is exclusively designed for the purpose of model fitting and sampling - opposite to the self-contained ``ROOT``. In other words, there is not attempt to extend its functionalities to features such as statistical methods (e.g. Upper limits/confidence intervals) or plotting libraries. A simple comparison can be driven by examining maximum likelihood fits. While ``zfit`` works as a backend for likelihood fits and can be integrated to packages such as ``lauztat`` and ``matplotlib``, ``RooFit`` performs the fit, the statistical treatment and plotting within. However, this also implies a limit scope of ``RooFit`` with respect to new minimisers, statistic methods and, broadly speaking, any new tool that might come. In summary, ``zfit`` aims to have a well stablished backend package in the context of an open source library. 
  
  * Another paramount aspect of ``zfit`` is its design for optimal parallelisation and scalability. Despite that the choice of ``TensorFlow`` as backend introduces a software dependence to the library, its use provides several features with interesting outcomes in the context of model fitting. The key concept is that ``TensorFlow`` is built under the dataflow programming model, specifically for parallel computations. In a simple way, ``TensorFlow`` creates a computational graph with the operations as the nodes of the graph and tensors to its edges. Hence, the computation only happens when the graph is executed in a session, which simplifies the parallelisation by identifying the dependencies between the edges and operations or even the partition across multiple devices (CPUs, GPUs). The architecture of ``zfit`` is built upon this idea and it aims to provide a high level interface to these features, i.e. most of the operations of graphs and evaluations are hidden for the user and it is designed to provide a natural and friendly model fitting and sampling experience. 



.. toctree::
    :titlesonly:
    :maxdepth: 1
    :caption: Contents:


    getting_started
    space
    parameter
    model
    data
    loss
    minimize
    API



