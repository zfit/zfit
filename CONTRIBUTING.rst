.. highlight:: console

=================
How to contribute
=================

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

* You can report bugs at https://github.com/zfit/zfit/issues.
* You can send feedback by filing an issue at https://github.com/zfit/zfit/issues or,
  for more informal discussions, you can also join our `Gitter channel <https://gitter.im/zfit/zfit>`_.


Get Started!
------------

Ready to contribute? Here's how to set up *zfit* for local development.

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) the *zfit* repo on GitHub.
2. Clone your fork locally:

.. code-block::

    git clone git@github.com:your_name_here/zfit.git

3. Synchronise your fork with the official *zfit* tags:

.. code-block::

    cd zfit
    git remote add upstream git@github.com:zfit/zfit.git
    git fetch upstream --tags

4. Install your local copy into a conda/mamba or other virtual environment. For example with conda, do

.. code-block::

    conda create -n zfit312 python=3.12
    conda activate zfit312
    python -m pip install -e .[dev]  # . is the the folder where pyproject.toml is 

Further, you can install the *zfit* pre-commit hooks locally to run the checks before you commit:

.. code-block::

    pre-commit install  # this will execute pre-commit checks before every commit

5. Create a branch for local development

.. code-block::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests (this can take a while ~30 mins). You can run the tests in parallel by
   installing ``pytest-xdist`` and running ``pytest -n NUM`` where ``NUM`` is the number of cores

.. code-block::

    pytest  # in the root folder of the repository where tests folder is

(Some tests have a ``@flaky`` decorator, which means that they might fail sometimes. If you see a flaky test
failing, inspect it, but most likely it's not a problem with your changes. Rerunning the specific test usually solves the problem.)



Some elements, like objects that can be dumped to HS3, new PDFs etc, have a ``truth``. This is a reference file and test that check against such a file are therefore expected to fail (as the truth file doesn't exist yet).
To create/regenerate a truth file, run the specific test that uses a truth file with the option ``--recreate-truth`` *after you verified that the output is actually correct*.

For example, to run the test ``test_dumpload_hs3_pdf`` in file ``tests/serialize/test_hs3_user.py``, run

.. code-block::

    pytest tests/serialize/test_hs3_user.py::test_dumpload_hs3_pdf -- --recreate-truth

7. Commit your changes and push your branch to GitHub

.. code-block::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website. The test suite is going
   to run again, testing all the necessary Python versions.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs may need to be updated. Put
   your new functionality into a function with a docstring (and add the
   necessary explanations in the corresponding rst file in the docs).
   If any math is involved, please document the exact formulae implemented
   in the docstring/docs.
   New elements, such as a new PDF, should for example be added to the
   ``docs/user_api/pdf/suitable_file.rst``.
