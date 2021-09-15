
.. |zfit_logo| image:: images/zfit-logo_400x168.png
   :target: https://github.com/zfit/zfit/
   :alt: zfit logo

.. |scikit-hep_logo| image:: images/scikit-hep-logo_168x168.png
   :target: https://scikit-hep.org/affiliated
   :alt: scikit-hep logo



|zfit_logo|

.. toctree::
    :maxdepth: 2
    :hidden:

    whats_new/index
    getting_started/index
    tutorials/index
    user_api/index
    project/index
    ask_a_question

=========================
Scalable pythonic fitting
=========================

.. link-button:: https://zfit-tutorials.readthedocs.io/en/latest/
    :type: url
    :text: Online interactive tutorials
    :classes: btn-outline-primary btn-block



.. panels::
    :header: text-center

    .. dropdown:: Getting started

        .. link-button:: getting_started/installation
            :type: ref
            :text: Installation
            :classes: btn-outline-primary btn-block

        .. link-button:: getting_started/5_minutes_to_zfit
            :type: ref
            :text: 5 minutes to zfit
            :classes: btn-outline-primary btn-block

        .. link-button:: ask_a_question
            :type: ref
            :text: Asking questions
            :classes: btn-outline-primary btn-block

    ---

    .. dropdown:: Tutorials

        .. link-button:: tutorials/introduction
            :type: ref
            :text: Introduction
            :classes: btn-outline-primary btn-block

        .. link-button:: https://zfit-tutorials.readthedocs.io/en/latest/
            :type: url
            :text: Interactive tutorials
            :classes: btn-outline-primary btn-block

        .. link-button:: tutorials/components/index
            :type: ref
            :text: Components
            :classes: btn-outline-primary btn-block

    ---

    .. link-button:: whats_new/index
        :type: ref
        :text: What's new?
        :classes: btn-outline-primary btn-block

    ---

    .. link-button:: user_api/index
        :type: ref
        :text: API documentation
        :classes: btn-outline-primary btn-block




The zfit package is a model fitting library based on `TensorFlow <https://www.tensorflow.org/>`_ and optimised for simple and direct manipulation of probability density functions. The main focus is on the scalability, parallelisation and a user friendly experience framework (no cython, no C++ needed to extend). The basic idea is to offer a pythonic oriented alternative to
the very successful RooFit library from the `ROOT <https://root.cern.ch/>`_ data analysis package. While RooFit has provided a stable platform for most of the needs of the High Energy Physics (HEP) community in the last few years, it has become increasingly difficult to integrate all the developments in the scientific Python ecosystem into RooFit due to its monolithic nature. Conversely, the core of zfit aims at becoming a solid ground for model fitting while providing enough flexibility to incorporate state-of-art tools and to allow scalability going to larger datasets.
This challenging task is tackled by following two basic design pillars:

- The skeleton and extension of the code is minimalist, simple and finite:
  the zfit library is exclusively designed for the purpose of model fitting and sampling---opposite to the self-contained RooFit/ROOT frameworks---with no attempt to extend its functionalities to features such as statistical methods or plotting.
  This design philosophy is well exemplified by examining maximum likelihood fits: while zfit works as a backend for likelihood fits and can be integrated to packages such as `hepstats <https://github.com/scikit-hep/hepstats>`_ and `matplotlib <https://matplotlib.org/>`_, RooFit performs the fit, the statistical treatment and plotting within.
  This wider scope of RooFit results in a lack of flexibility with respect to new minimisers, statistic methods and, broadly speaking, any new tool that might come.

- Another paramount aspect of zfit is its design for optimal parallelisation and scalability. Even though the choice of TensorFlow as backend introduces a strong software dependency, its use provides several interesting features in the context of model fitting.
  The key concept is that TensorFlow is built under the `dataflow programming model <https://en.wikipedia.org/wiki/Dataflow_programming>`_.
  Put it simply, TensorFlow creates a computational graph with the operations as the nodes of the graph and tensors to its edges. Hence, the computation only happens when the graph is executed in a session, which simplifies the parallelisation by identifying the dependencies between the edges and operations or even the partition across multiple devices (more details can be found in the `TensorFlow guide <https://www.tensorflow.org/guide/>`_).
  The architecture of zfit is built upon this idea and it aims to provide a high level interface to these features, *i.e.*, most of the operations of graphs and evaluations are hidden for the user, leaving a natural and friendly model fitting and sampling experience.

The zfit package is Free software, using an Open Source license. Both the software and this document are works in progress.
Source code can be found in `our github page <https://github.com/zfit/zfit/>`_.




.. hint::
    Yay, the new website design is here!

    We are proud to present the new look and feel of our website. While most of the
    things are where they used to be, here's a short explanation of the sections:

    * **What's new?**: Changelog and other new features of zfit.
    * **Getting started**: Installation guide, quickstart and examples.
    * **API reference**: Dive deep into the API.
    * **Project**: Learn who wrote zfit, how to contribute and other information about the project.
    * **Ask a question**: Does exactly what it says on the tin.

    If you have suggestions, contact us on our `Gitter channel`_ or open an issue on `GitHub`_.

    Thanks to pandas for open sourcing `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/en/latest/>`_.


|
