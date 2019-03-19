Welcome to zfit's documentation!
======================================

The ``zfit`` package is a model fitting library based on TensorFlow and optimised for simple and direct manipulation of probability density functions. The main focus is on the scalability, parallelisation and a user friendly experience framework (no cython, no C++ needed to extend). The basic idea is to provide a pythonic oriented alternative to 
the very successful ``RooFit`` libray from the ROOT (or even pyROOT) data analysis package. While ``RooFit`` has provided the a stable plattform for most of the needs of the High Energy Physics (HEP) community in the last few years, there are several developments in the machine learning (ML) community that have not been followed by this library. In this sense, the core of ``zfit`` is designed to provide this interface between the solid structure of ``RooFit`` with the state-of-art tools of ML. This challenge task can only be achieved with two paramount ansatz:   

  * The skeleton and extension of the code is minimalist/simple and finite. This is conceptually different from ``ROOT/RooFit`` philosophies, *i.e.* the ``zfit`` library is exclusively designed for the purpose of model fitting and sampling - opposite to the self-contained ``ROOT``. In other words, there is not attempt to extend its funcionalities to features such as statistical methods (e.g. Upper limits/confidence intervals) or plotting libraries. A simple comparison can be driven by examining maximum likelihood fits. While ``zfit`` works as a backend for likelihood fits and can be integrated to packages such as ``lauztat`` and ``matplotlib``, ``RooFit`` performs the fit, the statistical treament and plotting within. However, this also implies a limit scope of ``RooFit`` with respect to new minimisers, statistic methods and, broadly speaking, any new tool that might come. In summary, ``zfit`` aims to have a well stablished backend package in the context of an open source library. 
  
  * Another paramount aspect of ``zfit`` is the 
  
  
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



