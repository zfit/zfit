.. highlight:: shell

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

1. Fork the *zfit* repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/zfit.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv zfit
    $ cd zfit/
    $ pip install -e .[alldev]  # (or [dev] if this fails)

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the
   tests (this can take a while ~30 mins)::

    $ pytest


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website. The test suite is going
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
3. The pull request should work for all Python versions. Check
   https://travis-ci.org/zfit/zfit/pull_requests
   and make sure that the tests pass for all supported Python versions.
