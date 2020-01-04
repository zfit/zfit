============================
Downloading and Installation
============================

Prerequisites
-------------

``zfit`` works with Python versions 3.6 and 3.7.
The following packages are required:

- `tensorflow <https://www.tensorflow.org/>`_ >= 1.10.0
- `tensorflow_probability <https://www.tensorflow.org/probability>`_ >= 0.3.0
- `scipy <https://www.scipy.org/>`_ >=1.2
- `numpy <http://www.numpy.org/>`_
- `uproot <https://github.com/scikit-hep/uproot>`_
- `iminuit <https://github.com/scikit-hep/iminuit>`_
- `typing <https://docs.python.org/3/library/typing.html>`_
- `colorlog <https://github.com/borntyping/python-colorlog>`_
- `texttable <https://github.com/bufordtaylor/python-texttable>`_

All of these are readily available on PyPI, and should be installed automatically if installing with ``pip install zfit``.

In order to run the test suite, the `pytest <https://docs.pytest.org/en/latest/>`_ package is required

Downloads
---------
The latest beta version is available from `PyPi <https://pypi.org/project/zfit/>`.

Installation
------------

The easiest way to install zfit is with

.. code-block:: shell

   pip install zfit

To get the latest development version, use:

.. code-block:: shell

   git clone https://github.com/zfit/zfit.git

and install using:

.. code-block:: shell

   python setup.py install


Testing
-------

A battery of tests scripts that can be run with the `pytest <https://docs.pytest.org/en/latest/>`_ testing framework is distributed with zfit in the ``tests`` folder.
These are automatically run as part of the development process.
For any release or any master branch from the git repository, running ``pytest`` should run all of these tests to completion without errors or failures.

Getting help
------------

If you have questions, comments, or suggestions for zfit, please drop is a line in our `Gitter channel <https://gitter.im/zfit/zfit>`_.
If you find a bug in the code or documentation, open a `Github issue <https://github.com/zfit/zfit/issues/new>`_ and submit a report.
If you have an idea for how to solve the problem and are familiar with Python and GitHub, submitting a GitHub Pull Request would be greatly appreciated.
If you are unsure whether to use the Gitter channel or the Issue tracker, please start a conversation in the Gitter channel.

Acknowledgements
----------------

.. include:: ../THANKS.rst

License
-------

The zfit code is distributed under the BSD-3-Clause License:

.. include:: ../LICENSE

