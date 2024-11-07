*********
Changelog
*********

.. _newest-changelog:

0.23.0 (6 Nov. 2024)
======================

Major Features and Improvements
-------------------------------
- Minimizers can use the new ``SimpleLoss.from_any`` method that allows other libraries to hook into the minimization.
  For example, using zfit-physics, minimizers can directly minimize RooFit ``RooNllVar`` (as created by ``createNLL`` described `here <https://root.cern.ch/doc/master/classRooAbsPdf.html#a24b1afec4fd149e08967eac4285800de>`_
- Added the well performing ``LevenbergMarquardt`` minimizer, a new implementation of the Levenberg-Marquardt algorithm.
- New BFGS minimizer implementation of Scipy, ``ScipyBFGS``.
- Reactivate a few minimizers: ``ScipyDogleg``, ``ScipyNCG``, ``ScipyCOBYLA`` and ``ScipyNewtonCG``

Breaking changes
------------------
- removed multiple, old deprecated methods and arguments


Deprecations
-------------
- use ``stepsize`` instead of ``step_size`` in the ``zfit.Parameter`` constructor

Bug fixes and small changes
---------------------------
- add possibility to not jit by using ``force_eager`` in ``tf.function`` or raise a ``z.DoNotCompile`` error
- ``SimpleLoss`` can now be added together with another SimpleLoss
- ``get_params`` supports an ``autograd`` argument to filter parameters that do not support automatic differentiation.
  An object with parameters can advertise, which parameters are differentiable (with ``autograd_params``); by default, all
  parameters are assumed to be differentiable, the same effect as ``True``. If autograd is performed on parameters that
  do not support it, an error is raised.
- Use ``kanah`` sum for larger likelihoods by default to improve numerical stability
- Using the same ``zfit.Parameter`` for multiple arguments (i.e. to specify a common width in a PDF with a different width
  for left and right) could cause a crash due to some internal caching. This is now fixed.
- Minimizers have now been renamed without the trailing ``V1``. The old names are still available but will be removed in the future.

Experimental
------------

Requirement changes
-------------------

Thanks
------


0.22.0 22 Aug 2024
===================

Bug fixes and small changes
---------------------------
- change the truncated PDF with a yield to reflect a dynamic change in shape

Requirement changes
-------------------
- Upgrade from Pydantic V1 to V2


0.21.1 (13 Jul 2024)
========================

Bug fixes and small changes
---------------------------
- ``full`` argument for binned NLLs was not working properly and return a partially optimized loss value.
- jit compile all methods of the loss (gradient, hessian) to avoid recompilation every time. This can possibly speed up
  different minimizers significantly.



0.21.0 (2 Jul 2024)
========================

Major Features and Improvements
-------------------------------
- add ``JohnsonSU`` PDF, the Johnson SU distribution.



Bug fixes and small changes
---------------------------
- increase reliability of ``zfit.dill.dump`` and ``zfit.dill.dumps`` with an additional ``verify`` argument that reloads the dumped object to verify it was correctly dumped and retries if it wasn't.
- fix missing imported namespaces
- fixed a memory leak when creating multiple parameters
- add data loaders to ``zfit.data`` namespace



Requirement changes
-------------------
- upgrade to TensorFlow 2.17 and TensorFlow Probability 0.25

Thanks
------
- Davide Lancierini for finding and helping to debug the dill dumping issue
- James Herd for finding and reproducing the memory leak

0.20.3 (19 Apr 2024)
========================

Bug fixes and small changes
---------------------------
- consistent behavior in loss: simple loss can take a gradient and hesse function and the default base loss provides fallbacks that work correctly between ``value_gradient`` and ``gradient``. This maybe matters if you've implemented a custom loss and should fix any issues with it.
- multiprocessing would get stuck due to an `upstream bug in TensorFlow <https://github.com/tensorflow/tensorflow/issues/66115>`_. Working around it by disabling an unused piece of code.

Thanks
------
- acampoverde for finding the bug in the multiprocessing

0.20.2 (16 Apr 2024)
========================

Two small bugfixes
- fix backwards incompatible change of sampler
- detect if a RegularBinning has been transformed, raise error.

0.20.1 (14 Apr 2024)
========================

Major Features and Improvements
-------------------------------
- fix dumping and add convenience wrapper ``zfit.dill`` to dump and load objects with dill (a more powerful pickle). This way, any zfit object can be saved and loaded, such as ``FitResult`` that contains all other important objects to recreate the fit.
- improved performance for numerical gradient calculation, fixing also a minor numerical issue.

Bug fixes and small changes
---------------------------
- runing binned fits without a graph could deadlock, fixed.


0.20.0 (12 Apr 2024)
========================

Complete overhaul of zfit with a focus on usability and a variety of new pdfs!


Major Features and Improvements
-------------------------------
- Parameter behavior has changed, multiple parameters with the same name can now coexist!
  The ``NameAlreadyTakenError`` has been successfully removed (yay!). The new behavior only enforces that
  names and matching parameters *within a function/PDF/loss* are unique, as otherwise inconsistent expectations appear (for the `full discussion <https://github.com/zfit/zfit/discussions/342>`_).
- ``Space`` and limits have a complete overhaul in front of them, in short, these overcomplicated objects get simplified and the limits become more usable, in terms of dimensions. The `full discussion and changes can be found here <https://github.com/zfit/zfit/discussions/533>`_ .
- add an unbinned ``Sampler`` to the public namespace under ``zfit.data.Sampler``: this object is returned in the ``create_sampler`` method and allows to resample from a function without recreating the compiled function, i.e. loss. It has an additional method ``update_data`` to update the data without recompiling the loss and can be created from a sample only. Useful to have a custom dataset in toys.
- allow to use pandas DataFrame as input where zfit Data objects are expected
- Methods of PDFs and loss functions that depend on parameters take now the value of a parameter explicitly as arguments, as a mapping of str (parameter name) to value.
- Python 3.12 support
- add ``GeneralizedCB`` PDF which is similar to the ``DoubleCB`` PDF but with different standard deviations for the left and right side.
- Added functor for PDF caching ``CachedPDF``: ``pdf``, ``integrate`` PDF methods can be cacheable now
- add ``faddeeva_humlicek`` function under the ``zfit.z.numpy`` namespace. This is an implementation of the Faddeeva function, combining Humlicek's rational approximations according to Humlicek (JQSRT, 1979) and Humlicek (JQSRT, 1982).
- add ``Voigt`` profile PDF which is a convolution of a Gaussian and a Cauchy distribution.
- add ``TruncatedPDF`` that allows to truncate in one or multiple ranges (replaces "MultipleLimits" and "MultiSpace")
- add ``LogNormal`` PDF, a log-normal distribution, which is a normal distribution of the logarithm of the variable.
- add ``ChiSquared`` PDF, the standard chi2 distribution, taken from `the tensorflow-probability implementation <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi2>`_.
- add ``StudentT`` PDF, the standard Student's t distribution, taken from `tensorflow-probability implementation <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentT>`_.
- add ``GaussExpTail`` and ``GeneralizedGaussExpTail`` PDFs, which are a Gaussian with an exponential tail on one side and a Gaussian with different sigmas on each side and different exponential tails on each side respectively.
- add ``QGauss`` PDF, a distribution that arises from the maximization of the Tsallis entropy under appropriate constraints, `see here <https://en.wikipedia.org/wiki/Q-Gaussian_distribution>`_.
- add ``BifurGauss`` PDF, a Gaussian distribution with different sigmas on each side of the mean.
- add ``Bernstein`` PDF, which is a PDF defined by a linear combination of Bernstein polynomials given their coefficients.
- add ``Gamma`` PDF, the Gamma distribution.
- ``Data`` has now a ``with_weights`` method that returns a new data object with different weights and an improved ``with_obs`` that allows to set obs with new limits. These replace the ``set_weights`` and ``set_data_range`` methods for a more functional approach.
- add ``label`` to different objects (PDF, Data, etc.) that allows to give a human-readable name to the object. This is used in the plotting and can be used to identify objects.
  Notably, Parameters have a label that can be arbitrary. ``Space`` has one label for each observable if the space is a product of spaces. ``Space.label`` is a string and only possible for one-dimensional spaces, while ``Space.labels`` is a list of strings and can be used for any, one- or multi-dimensional spaces.
- add ``zfit.data.concat(...)`` to concatenate multiple data objects into one along the index or along the observables. Similar to ``pd.concat``.
- PDFs now have a ``to_truncated`` method that allows to create a truncated version of the PDF, possibly with different and multiple limits. This allows to easily create a PDF with disjoint limits.
- ``Data`` and ``PDF`` that take ``obs`` in the initialization can now also take binned observables, i.e. a ``zfit.Space`` with ``binning=...`` and will return a binned version of the object (``zfit.data.BinnedData`` or ``zfit.pdf.BinnedFromUnbinned``, where the latter is a generic wrapper). This is equivalent of calling ``to_binned`` on the objects)
- ``zfit.Data`` can be instantiated directly with most data types, such as numpy arrays, pandas DataFrames etc insead of using the dedicated constructors ``from_numpy``, ``from_pandas`` etc.
  The constructors may still provide additional functionality, but overall, the switch should be seamless.


Breaking changes
------------------
This release contains multiple "breaking changes", however, the vast majority if not all apply only for edge cases and undocummented functions.

- a few arguments are now keyword-only arguments. This *can* break existing code if the arguments were given as positional arguments. Just use the appropriate keyword arguments instead.
  (Example: instead of using ``zfit.Space(obs, limits)`` use ``zfit.Space(obs, limits=limits)``).
  This was introduced to make the API more robust and to avoid errors due to the order of arguments, with a few new ways of creating objects.
- ``Data.from_root``: deprecated arguments ``branches`` and ``branch_aliases`` have been removed. Use ``obs`` and ``obs_aliases`` instead.
- ``NameAlreadyTakenError`` was removed, see above for the new behavior. This should not have an effect on any existing code *except if you relied on the error being thrown*.
- Data objects had an intrinsic, TensorFlow V1 legacy behavior: they were actually cut when the data was *retrieved*. This is now changed and the data is cut when it is created. This should not have any impact on existing code and just improve runtime and memory usage.
- Partial integration used to use some broadcasting tricks that could potentially fail. It uses now a dynamic while loop that _could_ be slower but works for arbitrary PDFs. This should not have any impact on existing code and just improve stability (but technically, the data given to the PDF *if doing partial integration* is now "different", in the sense that it's now not different anymore from any other call)
- if a ``tf.Variable`` was used to store the number of sampled values in a sampler, it was possible to change the value of that variable to change the number of samples drawn. This is now not possible anymore and the number of samples should be given as an argument ``n`` to the ``resample`` method, as was possible since a long time.
- ``create_sampler`` has a breaking change for ``fixed_params``: when the argument was set to False, any change in the parameters would be reflected when resampling.
  This highly statebased behavior was confusing and is now removed. The argument is now called ``params``
  and behaves as expected: the sampler will remember the parameters at the time of creation,
  possibly updated with ``params`` and will not change anymore. To sample from a different set of parameters,
  the params have to be passed to the ``resample`` method _explicitly_.
- the default names for ``hesse`` and ``errors`` have now been changed to ``hesse`` and ``errors``, respectively.
  This was deprecated since a while and both names were available for backwards compatibility. The old names are now removed. If you get an error, ``minuit_hessse`` or ``minuit_minos`` not found, just replace it with ``hesse`` and ``errors``.



Deprecations
-------------
- ``result.fminfull`` is deprecated and will be removed in the future. Use ``result.fmin`` instead.
- ``Data.set_data_range`` is deprecated and will be removed in the future. Use ``with_range`` instead.
- ``Space`` has many deprecated methods, such as ``rect_limits`` and quite a few more. The full discussion can be found `here <https://github.com/zfit/zfit/discussions/533>`_.
- ``fixed_params`` in ``create_sampler`` is deprecated and will be removed in the future. Use ``params`` instead.
- ``fixed_params`` attribute of the ``Sampler`` is deprecated and will be removed in the future. Use ``params`` instead.
- ``uncertainties`` in ``GaussianConstraint`` is deprecated and will be removed in the future. Use either explicitly ``sigma`` or ``cov``.
- the ``ComposedParameter`` and ``ComplexParameter`` argument ``value_fn`` is deprecated in favor of the new argument ``func``. Identical behavior.
- ``zfit.run(...)`` is deprecated and will be removed in the future. Simply remove it should work in most cases.
  (if an explicity numpy, not just array-like, cast is needed, use ``np.asarray(...)``. But usually this is not needed). This function is an old relic from the past TensorFlow 1.x, ``tf.Session`` times and is not needed anymore. We all remember well these days :)

Bug fixes and small changes
---------------------------
- complete overhaul of partial integration that used some broadcasting tricks that could potentially fail. It uses now a dynamic while loop that _could_ be slower but works for arbitrary PDFs and no problems should be encountered anymore.
- ``FitResult`` can now be used as a context manager, which will automatically set the values of the parameters to the best fit values and reset them to the original values after the context is left. A new method ``update_params`` allows to update the parameters with the best fit values explicitly.
- ``result.fmin`` now returns the full likelihood, while ``result.fminopt`` returns the optimized likelihood with potential constant subtraction. The latter is mostly used by the minimizer and other libraries. This behavior is consistent with the behavior of other methods in the loss that return by default the full, unoptimized value.
- serialization only allowed for one specific limit (space) of each obs. Multiple, independent
  limits can now be serialized.
- Increased numerical stability: this was compromised due to some involuntary float32 conversions in TF. This has been fixed.
- arguments ``sigma`` and ``cov`` are now used in ``GaussianConstraint``, both mutually exclusive, to ensure the intent is clear.
- improved hashing and precompilation in loss, works now safely also with samplers.
- seed setting is by default completely randomized. This is a change from the previous behavior where the seed was set to a more deterministic value. Use seeds only for reproducibility and not for real randomness, as some strange correlations between seeds have been observed. To guarantee full randomness, just call ``zfit.run.set_seed()`` without arguments.
- ``zfit.run.set_seed`` now returns the seed that was set. This is useful for reproducibility.

Experimental
------------

- a simple ``plot`` mechanism has been added with ``pdf.plot.plotpdf`` to plot PDFs. This is simple and fully interacts with matplotlib, allowing to plot quickly in a more interactive way.
- ``zfit.run.experimental_disable_param_update``: this is an experimental feature that allows to disable the parameter update in a fit as is currently done whenever ``minimize`` is called. In conjunction with the new method ``update_params()``, this can be used as ``result = minimizer.minimize(...).update_params()`` to keep the same behavior as currently. Also, the context manager of ``FitResult`` can be used to achieve the same behavior in a context manager (with minimizer.minimize(...) as result: ...) also works.

Requirement changes
-------------------
- upgrade to TensorFlow 2.16 and TensorFlow Probability 0.24

Thanks
------
- huge thanks to @ikrommyd (Iason Krommydas) for the addition of various PDFs and to welcome him on board as a new contributor!
- @anjabeck for the addition of the ``ChiSquared`` PDF

0.18.2 (13 Mar 2024)
========================

Hotfix for missing dependency attrs

0.18.1 (22 Feb 2024)
========================

Bug fixes in randomness and improved caching

Major Features and Improvements
-------------------------------

- reduced the number of graph caching reset, resulting in significant speedups in some cases


Bug fixes and small changes
---------------------------
 - use random generated seeds for numpy and TF, as they can otherwise have unwanted correlations


Thanks
------
@anjabeck for the bug report and the idea to use random seeds for numpy and TF
@acampoverde for reporting the caching issue

0.18.0 (29 Jan 2024)
========================

Major Features and Improvements
-------------------------------
- update to TensorFlow 2.15, TensorFlow Probability 0.23
- drop Python 3.8 support


0.17.0 (29 Jan 2024)
========================

TensorFlow 2.15, drop Python 3.8 support

Major Features and Improvements
-------------------------------
- add correct uncertainty for unbinned, weighted fits with constraints and/or that are extended.
- allow mapping in ``zfit.param.set_values`` for values


Bug fixes and small changes
---------------------------
- fix issues where EDM goes negative, set to 999
- improved stability of the loss evaluation
- improved uncertainty calculation accuracy with ``zfit_error``



Thanks
------

Daniel Craik for the idea of allowing a mapping in ``set_values``

0.16.0 (26 July 2023)
========================

Major Features and Improvements
-------------------------------

- add ``full`` option to loss call of ``value``, which returns the unoptimized value allowing for easier statistical tests using the loss.
  This is the default behavior and should not break any backwards compatibility, as the "not full loss" was arbitrary.
- changed the ``FitResult`` to print both loss values, the unoptimized (full) and optimized (internal)


Bug fixes and small changes
---------------------------
- bandwidth preprocessing was ignored in KDE
- ``unstack_x`` with an ``obs`` as argument did return the wrong shape


Thanks
------
@schmitse for reporting the bug in the KDE bandwidth preprocessing
@lorenzopaolucci for bringing up the absolute value of the loss in the fitresult as an issue

0.15.5 (26 July 2023)
========================

Bug fixes and small changes
---------------------------
- fix a bug in histmodifier that would not properly take into account the yield of the wrapped PDF

0.15.2 (20 July 2023)
========================

Fix missing ``attrs`` dependency

Major Features and Improvements
-------------------------------
- add option ``full`` in loss to return the full, unoptimized value (currently not default), allowing for easier statistical tests using the loss



0.15.0 (13 July 2023)
========================

Update to TensorFlow 2.13.x

Requirement changes
-------------------
- TensorFlow upgraded to ~=2.13.0
- as TF 2.13.0 ships with the arm64 macos wheels, the requirement of ``tensorflow_macos`` is removed

Thanks
------
- Iason Krommydas for helping with the macos requirements for TF

0.14.1 (1 July 2023)
========================

Major Features and Improvements
-------------------------------

- zfit broke for pydantic 2, which upgraded.


Requirement changes
-------------------
- restrict pydantic to <2.0.0

0.14.0 (22 June 2023)
========================

Major Features and Improvements
-------------------------------

- support for Python 3.11, dropped support for Python 3.7

Bug fixes and small changes
---------------------------
-fix longstanding bug in parameters caching


Requirement changes
-------------------
- update to TensorFlow 2.12
- removed ``tf_quant_finance``


0.13.2 (15. June 2023)
========================

Bug fixes and small changes
---------------------------
- fix a caching problem with parameters (could cause issues with larger PDFs as params would be "remembered" wrongly)
- more helpful error message when jacobian (as used for weighted corrections) is analytically asked but fails
- make analytical gradient for CB integral work


0.13.1 (20 Apr 2023)
========================

Bug fixes and small changes
---------------------------
- array bandwidth for KDE works now correctly

Requirement changes
-------------------
- fixed uproot for Python 3.7 to <5

Thanks
------
- @schmitse for reporting and solving the bug in the KDE bandwidth with arrays

0.13.0 (19 April 2023)
========================

Major Features and Improvements
-------------------------------

last Python 3.7 version

Bug fixes and small changes
---------------------------
- ``SampleData`` is not used anymore, a ``Data`` object is returned (for simple sampling). The ``create_sampler`` will still return a ``SamplerData`` object though as this differs from ``Data``.

Experimental
------------
- Added support on a best-effort for human-readable serialization of objects including an HS3-like representation, find a `tutorial on serialization here <https://zfit-tutorials.readthedocs.io/en/latest/tutorials/components/README.html#serialization>`_. Most built-in unbinned PDFs are supported. This is still experimental and not yet fully supported. Dumping can be performed safely, loading maybe easily breaks (also between versions), so do not rely on it yet. Everything else - apart of trying to dump - should only be used for playing around and giving feedback purposes.

Requirement changes
-------------------
- allow uproot 5 (remove previous restriction)

Thanks
------
- to Johannes Lade for the amazing work on the serialization, which made this HS3 implementation possible!


0.12.1 (1 April 2023)
========================


Bug fixes and small changes
---------------------------
- added ``extended`` as a parameter to all PDFs: a PDF can now directly be extended without the need for
  ``create_extended`` (or ``set_yield``).
- ``to_pandas`` and ``from_pandas`` now also support weights as columns. Default column name is ``""``.
- add ``numpy`` and ``backend`` to options when setting the seed
- reproducibility by fixing the seed in zfit is restored, ``zfit.run.set_seed`` now also sets the seed for the backend(numpy, tensorflow, etc.) if requested (on by default)

Thanks
------
- Sebastian Schmitt @schmitse for reporting the bug in the non-reproducibility of the seed.

0.12.0 (13 March 2023)
========================

Bug fixes and small changes
---------------------------
- ``create_extended`` added ``None`` to the name, removed.
- ``SimpleConstraint`` now also takes a function that has an explicit ``params`` argument.
- add ``name`` argument to ``create_extended``.
- adding binned losses would error due to the removed ``fit_range`` argument.
- setting a global seed made the sampler return constant values, fixed (unoptimized but correct). If you ran
  a fit with a global seed, you might want to rerun it.
- histogramming and limit checks failed due to a stricter Numpy check, fixed.


Thanks
------
- @P-H-Wagner for finding the bug in ``SimpleConstraint``.
- Dan Johnson for finding the bug in the binned loss that would fail to sum them up.
- Hanae Tilquin for spotting the bug with TensorFlows changed behavior or random states inside a tf.function,
  leading to biased samples whenever a global seed was set.

0.11.1 (20 Nov 2022)
=========================

Hotfix for wrong import

0.11.0 (29 Nov 2022)
========================

Major Features and Improvements
-------------------------------
- columns of unbinned ``data`` can be accessed with the obs like a mapping (like a dataframe)
- speedup builtin ``errors`` method and make it more robust

Breaking changes
------------------
- ``Data`` can no longer be used directly as an array-like object but got mapping-like behavior.
- some old deprecated methods were removed

Bug fixes and small changes
---------------------------
- improved caching speed, reduced tradeoff against memory
- yields were not added correctly in some (especially binned) PDFs and the fit would fail

Requirement changes
-------------------
- add jacobi (many thanks at @HansDembinski for the package)


0.10.1 (31 Aug 2022)
========================

Major Features and Improvements
-------------------------------
- reduce the memory footprint on (some) fits, especially repetitive (loops) ones.
  Reduces the number of cached compiled functions. The cachesize can be set with
  ``zfit.run.set_cache_size(int)``
  and specifies the number of compiled functions that are kept in memory. The default is 10, but
  this can be tuned. Lower values can reduce memory usage, but potentially increase runtime.


Bug fixes and small changes
---------------------------
- Enable uniform binning for n-dimensional distributions with integer(s).
- Sum of histograms failed for calling the pdf method (can be indirectly), integrated over wrong axis.
- Binned PDFs expected binned spaces for limits, now unbinned limits are also allowed and automatically
    converted to binned limits using the PDFs binning.
- Speedup sampling of binned distributions.
- add ``to_binned`` and ``to_unbinned`` methods to PDF


Thanks
------
- Justin Skorupa for finding the bug in the sum of histograms and the missing automatic
  conversion of unbinned spaces to binned spaces.

0.10.0 (22. August 2022)
========================

Public release of binned fits and upgrade to Python 3.10 and TensorFlow 2.9.

Major Features and Improvements
-------------------------------
- improved data handling in constructors ``from_pandas`` (which allows now to
  have weights as columns, dataframes that are a superset of the obs) and
  ``from_root`` (obs can now be spaces and therefore cuts can be direcly applied)
- add hashing of unbinned datasets with a ``hashint`` attribute. None if no hash was possible.

Breaking changes
------------------


Deprecations
-------------

Bug fixes and small changes
---------------------------
- SimpleLoss correctly supports both functions with implicit and explicit parameters, also if they
  are decorated.
- extended sampling errored for some cases of binned PDFs.
- ``ConstantParameter`` errored when converted to numpy.
- Simultaneous binned fits could error with different binning due to a missing sum over
  a dimension.
- improved stability in loss evaluation of constraints and poisson/chi2 loss.
- reduce gradient evaluation time in ``errors`` for many parameters.
- Speedup Parameter value assignement in fits, which is most notably when the parameter update time is
  comparably large to the fit evaluation time, such as is the case for binned fits with many nuisance
  parameters.
- fix ipyopt was not pickleable in a fitresult
- treat parameters sometimes as "stateless", possibly reducing the number of retraces and reducing the
  memory footprint.

Experimental
------------

Requirement changes
-------------------
- nlopt and ipyopt are now optional dependencies.
- Python 3.10 added
- TensorFlow >= 2.9.0, <2.11 is now required and the corresponding TensorFlow-Probability
  version >= 0.17.0, <0.19.0

Thanks
------
- @YaniBion for discovering the bug in the extended sampling and testing the alpha release
- @ResStump for reporting the bug with the simultaneous binned fit

0.9.0a2
========

Major Features and Improvements
-------------------------------
- Save results by pickling, unpickling a frozen (``FitResult.freeze()``) result and using
  ``zfit.param.set_values(params, result)`` to set the values of ``params``.



Deprecations
-------------
- the default name of the uncertainty methods ``hesse`` and ``errors`` depended on
  the method used (such as ``"minuit_hesse"``, ``"zfit_errors"`` etc.) and would be the exact method name.
  New names are now 'hesse' and 'errors', independent of the method used. This reflects better that the
  methods, while internally different, produce the same result.
  To update, use 'hesse' instead of 'minuit_hesse' or 'hesse_np' and 'errors' instead of 'zfit_errors'
  or ``"minuit_minos"`` in order to access the uncertainties in the fitresult.
  Currently, the old names are still available for backwards compatibility.
  If a name was explicitly chosen in the error method, nothing changed.

Bug fixes and small changes
---------------------------
- KDE datasets are now correctly mirrored around observable space limits
- multinomial sampling would return wrong results when invoked multiple times in graph mode due to
  a non-dynamic shape. This is fixed and the sampling is now working as expected.
- increase precision in FitResult string representation and add that the value is rounded


Thanks
------
 - schmitse for finding and fixing a mirroring bug in the KDEs
 - Sebastian Bysiak for finding a bug in the multinomial sampling

0.9.0a0
========

Major Features and Improvements
-------------------------------

- Binned fits support, although limited in content, is here! This includes BinnedData, binned PDFs, and
  binned losses. TODO: extend to include changes/point to binned introduction.
- new Poisson PDF
- added Poisson constraint, LogNormal Constraint
- Save results by pickling, unpickling a frozen (``FitResult.freeze()``) result and using
  ``zfit.param.set_values(params, result)`` to set the values of ``params``.

Breaking changes
------------------

- params given in ComposedParameters are not sorted anymore. Rely on their name instead.
- ``norm_range`` is now called ``norm`` and should be replaced everywhere if possible. This will break in
  the future.

Deprecation
-------------

Bug fixes and small changes
---------------------------
- remove warning when using ``rect_limits`` or similar.
- gauss integral accepts now also tensor inputs in limits
- parameters at limits is now shown correctly

Experimental
------------

Requirement changes
-------------------
- add TensorFlow 2.7 support

Thanks
------


0.8.3 (5 Apr 2022)
===================
- fixate nlopt to < 2.7.1


0.8.2 (20 Sep 2021)
====================

Bug fixes and small changes
---------------------------
- fixed a longstanding bug in the DoubleCB implementation of the integral.
- remove outdated deprecations

0.8.1 (14. Sep. 2021)
======================

Major Features and Improvements
-------------------------------

- allow ``FitResult`` to ``freeze()``, making it pickleable. The parameters
  are replaced by their name, the objects such as loss and minimizer as well.
- improve the numerical integration by adding a one dimensional efficient integrator, testing for the accuracy of
  multidimensional integrals. If there is a sharp peak, this maybe fails to integrate and the number of points
  has to be manually raised
- add highly performant kernel density estimation (mainly contributed by Marc Steiner)
  in 1 dimension which allow
  for the choice of arbitrary kernels, support
  boundary mirroring of the data and allow for large (millions) of data samples:
  - :class:`~zfit.pdf.KDE1DimExact` for the normal density estimation
  - :class:`~zfit.pdf.KDE1DimGrid` using a binning
  - :class:`~zfit.pdf.KDE1DimFFT` using a binning and FFT
  - :class:`~zfit.pdf.KDE1DimISJ` using a binning and an algorithm (ISJ) to solve the optimal bandwidth

  For an introduction, see either :ref:`sec-kernel-density-estimation` or the tutorial :ref:`sec-components-model`

- add windows in CI

Breaking changes
------------------
- the numerical integration improved with more sensible values for tolerance. This means however that some fits will
  greatly increase the runtime. To restore the old behavior globally, do
  for each instance ``pdf.update_integration_options(draws_per_dim=40_000, max_draws=40_000, tol=1)``
  This will integrate regardless of the chosen precision and it may be non-optimal.
  However, the precision estimate in the integrator is also not perfect and maybe overestimates the error, so that
  the integration by default takes longer than necessary. Feel free to play around with the parameters and report back.


Bug fixes and small changes
---------------------------
- Double crystallball: move a minus sign down, vectorize the integral, fix wrong output shape of pdf
- add a minimal value in the loss to avoid NaNs when taking the log of 0
- improve feedback when taking the derivative with respect to a parameter that
  a function does not depend on or if the function is purely Python.
- make parameters deletable, especially it works now to create parameters in a function only
  and no NameAlreadyTakenError will be thrown.


Requirement changes
-------------------

- add TensorFlow 2.6 support (now 2.5 and 2.6 are supported)

Thanks
------
- Marc Steiner for contributing many new KDE methods!


0.7.2 (7. July 2021)
======================

Bug fixes and small changes
---------------------------
- fix wrong arguments to ``minimize``
- make BaseMinimizer arguments optional

0.7.1 (6. July 2021)
======================


Bug fixes and small changes
---------------------------
- make loss callable with array arguments and therefore combatible with iminuit cost functions.
- fix a bug that allowed FitResults to be valid that are actually invalid (reported by Maxime Schubiger).


0.7.0 (03 Jun 2021)
=====================

Major Features and Improvements
-------------------------------
- add Python 3.9 support
- upgrade to TensorFlow 2.5

Bug fixes and small changes
---------------------------
- Scipy minimizers with hessian arguments use now ``BFGS`` as default


Requirement changes
-------------------

- remove Python 3.6 support
- boost-histogram



0.6.6 (12.05.2021)
==================

Update ipyopt requirement < 0.12 to allow numpy compatible with TensorFlow

0.6.5 (04.05.2021)
==================

- hotfix for wrong argument in exponential PDF
- removed requirement ipyopt, can be installed with ``pip install zfit[ipyopt]``
  or by manually installing ``pip install ipyopt``



0.6.4 (16.4.2021)
==================


Bug fixes and small changes
---------------------------
- remove requirement of Ipyopt on MacOS as no wheels are available. This rendered zfit
  basically non-installable.


0.6.3 (15.4.2021)
==================


Bug fixes and small changes
---------------------------
- fix loss failed for large datasets
- catch hesse failing for iminuit


0.6.2
========

Minor small fixes.


Bug fixes and small changes
---------------------------

- add ``loss`` to callback signature that gives full access to the model
- add :meth:`~zfit.loss.UnbinnedNLL.create_new` to losses in order to re-instantiate
  them with new models and data
  preserving their current (and future) options and other arguments


0.6.1 (31.03.2021)
===================
Release for fix of minimizers that performed too bad

Breaking changes
------------------
- remove badly performing Scipy minimizers :class:`~zfit.minimize.ScipyTrustKrylovV1` and
  :class:`~zfit.minimize.ScipyTrustNCGV1`

Bug fixes and small changes
---------------------------
- fix auto conversion to complex parameter using constructor


0.6.0 (30.3.2021)
===================

Added many new minimizers from different libraries, all with uncertainty estimation available.

Major Features and Improvements
-------------------------------

- upgraded to TensorFlow 2.4
- Added many new minimizers. A full list can be found in :ref:`minimize_user_api`.

  - :class:`~zfit.minimize.IpyoptV1` that wraps the powerful Ipopt large scale minimization library
  - Scipy minimizers now have their own, dedicated wrapper for each instance such as
    :class:`~zfit.minimize.ScipyLBFGSBV1`, or :class:`~zfit.minimize.ScipySLSQPV1`
  - NLopt library wrapper that contains many algorithms for local searches such as
    :class:`~zfit.minimize.NLoptLBFGSV1`, :class:`~zfit.minimize.NLoptTruncNewtonV1` or
    :class:`~zfit.minimize.NLoptMMAV1` but also includes more global minimizers such as
    :class:`~zfit.minimize.NLoptMLSLV1` and :class:`~zfit.minimize.NLoptESCHV1`.

- Completely new and overhauled minimizers design, including:

  - minimizers can now be used with arbitrary Python functions and an initial array independent of zfit
  - a minimization can be 'continued' by passing ``init`` to ``minimize``
  - more streamlined arguments for minimizers, harmonized names and behavior.
  - Adding a flexible criterion (currently EDM) that will terminate the minimization.
  - Making the minimizer fully stateless.
  - Moving the loss evaluation and strategy into a LossEval that simplifies the handling of printing and NaNs.
  - Callbacks are added to the strategy.

- Major overhaul of the ``FitResult``, including:

  - improved ``zfit_error`` (equivalent of ``MINOS``)
  - ``minuit_hesse`` and ``minuit_minos`` are now available with all minimizers as well thanks to an great
    improvement in iminuit.
  - Added an ``approx`` hesse that returns the approximate hessian (if available, otherwise empty)

- upgrade to iminuit v2 changes the way it works and also the Minuit minimizer in zfit,
  including a new step size heuristic.
  Possible problems can be caused by iminuit itself, please report
  in case your fits don't converge anymore.
- improved ``compute_errors`` in speed by caching values and the reliability
  by making the solution unique.
- increased stability for large datasets with a constant subtraction in the NLL

Breaking changes
------------------
- NLL (and extended) subtracts now by default a constant value. This can be changed with a new ``options`` argument.
  COMPARISON OF DIFFEREN NLLs (their absolute values) fails now! (flag can be deactivated)
- BFGS (from TensorFlow Probability) has been removed as it is not working properly. There are many alternatives
  such as ScipyLBFGSV1 or NLoptLBFGSV1
- Scipy (the minimizer) has been removed. Use specialized ``Scipy*`` minimizers instead.
- Creating a ``zfit.Parameter``, usign ``set_value`` or ``set_values`` now raises a ``ValueError``
  if the value is outside the limits. Use ``assign`` to suppress it.

Deprecation
-------------
- strategy to minimizer should now be a class, not an instance anymore.

Bug fixes and small changes
---------------------------
- ``zfit_error`` moved only one parameter to the correct initial position. Speedup and more reliable.
- FFTconv was shifted if the kernel limits were not symetrical, now properly taken into account.
- circumvent overflow error in sampling
- shuffle samples from sum pdfs to ensure uniformity and remove conv sampling bias
- ``create_sampler`` now samples immediately to allow for precompile, a new hook that will allow objects to optimize
  themselves.


Requirement changes
-------------------
- ipyopt
- nlopt
- iminuit>=2.3
- tensorflow ~= 2.4
- tensorflow-probability~=12

For devs:
- pre-commit
- pyyaml
- docformatter


Thanks
------

- Hans Dembinski for the help on upgrade to imituit V2
- Thibaud Humair for helpful remarks on the parameters


0.5.6 (26.1.2020)
=================

Update to fix iminuit version

Bug fixes and small changes
---------------------------
- Fix issue when using a ``ComposedParameter`` as the ``rate`` argument of a ``Poisson`` PDF

Requirement changes
-------------------
- require iminuit < 2 to avoid breaking changes


0.5.5 (20.10.2020)
==================

Upgrade to TensorFlow 2.3 and support for weighted hessian error estimation.

Added a one dimensional Convolution PDF

Major Features and Improvements
-------------------------------

- upgrad to TensorFlow 2.3

Breaking changes
------------------

Deprecation
-------------

Bug fixes and small changes
---------------------------

- print parameter inside function context works now correctly

Experimental
------------

- Computation of the covariance matrix and hessian errors with weighted data
- Convolution PDF (FFT in 1Dim) added (experimental, feedback welcome!)

Requirement changes
-------------------

- TensorFlow==2.3 (before 2.2)
- ``tensorflow_probability==0.11``
- tensorflow-addons  # spline interpolation in convolution


Thanks
------



0.5.4 (16.07.2020)
==================


Major Features and Improvements
-------------------------------
- completely new doc design

Breaking changes
------------------
- Minuit uses its own, internal gradient by default. To change this back, use ``use_minuit_grad=False``
- ``minimize(params=...)`` now filters correctly non-floating parameters.
- ``z.log`` has been moved to ``z.math.log`` (following TF)


Bug fixes and small changes
---------------------------
- ncalls is not correctly using the internal heuristc or the ncalls explicitly
- ``minimize(params=...)`` automatically extracts independent parameters.
- fix copy issue of KDEV1 and change name to 'adaptive' (instead of 'adaptiveV1')
- change exp name of ``lambda_`` to lam (in init)
- add ``set_yield`` to BasePDF to allow setting the yield in place
- Fix possible bug in SumPDF with extended pdfs (automatically)

Experimental
------------

Requirement changes
-------------------
- upgrade to iminuit>=1.4
- remove cloudpickle hack fix

Thanks
------
Johannes for the docs re-design

0.5.3 (02.07.20)
================

Kernel density estimation for 1 dimension.

Major Features and Improvements
-------------------------------
- add correlation method to FitResult
- Gaussian (Truncated) Kernel Density Estimation in one dimension ``zfit.pdf.GaussianKDE1DimV1`` implementation with fixed and
  adaptive bandwidth added as V1. This
  is a feature that needs to be improved and feedback is welcome
- Non-relativistic Breit-Wigner PDF, called Cauchy, implementation added.

Breaking changes
------------------
- change human-readable name of ``Gauss``, ``Uniform`` and ``TruncatedGauss`` to remove the ``'_tfp'`` at the end of the name



Bug fixes and small changes
---------------------------
- fix color wrong in printout of results, params
- packaging: moved to pyproject.toml and a setup.cfg mainly, development requirements can
  be installed with the ``dev`` extra as (e.g.) ``pip install zfit[dev]``
- Fix shape issue in TFP distributions for partial integration
- change zfit internal algorithm (``zfit_error``) to compute error/intervals from the profile likelihood,
  which is 2-3 times faster than previous algorithm.
- add ``from_minuit`` constructor to ``FitResult`` allowing to create it when
  using directly iminuit
- fix possible bias with sampling using accept-reject

Requirement changes
-------------------
- pin down cloudpickle version (upstream bug with pip install) and TF, TFP versions


0.5.2 (13.05.2020)
==================


Major Features and Improvements
-------------------------------
- Python 3.8 and TF 2.2 support
- easier debugigng with ``set_graph_mode`` that can also be used temporarily
  with a context manager. False will make everything execute Numpy-like.

Bug fixes and small changes
---------------------------
- added ``get_params`` to loss
- fix a bug with the ``fixed_params`` when creating a sampler
- improve exponential PDF stability and shift when normalized
- improve accept reject sampling to account for low statistics


Requirement changes
-------------------

- TensorFlow >= 2.2

0.5.1 (24.04.2020)
==================
(0.5.0 was skipped)

Complete refactoring of Spaces to allow arbitrary function.
New, more consistent behavior with extended PDFs.
SumPDF refactoring, more explicit handling of fracs and yields.
Improved graph building allowing for more fine-grained control of tracing.
Stabilized minimization including a push-back for NaNs.



Major Features and Improvements
-------------------------------
- Arbitrary limits as well as vectorization (experimental)
  are now fully supported. The new ``Space`` has an additional argument for a function that
  tests if a vector x is inside.

  To test if a value is inside a space, ``Space.inside`` can be used. To filter values, ``Space.filter``.

  The limits returned are now by default numpy arrays with the shape ``(1, n_obs)``. This corresponds well
  to the old layout and can, using ``z.unstack_x(lower)`` be treated like ``Data``. This has also some
  consequences for the output format of ``rect_area``: this is now a vector.

  Due to the ambiguity of the name ``limits``, ``area`` etc (since they do only reflect the rectangular case)
  method with leading ``rect_*`` have been added (``rect_limits``, ``rect_area`` etc.) and are encouraged to be used.

- Extending a PDF is more straightforward and removes any "magic". The philosophy is: a PDF can be extended
  or not. But it does not change the fundamental behavior of functions.

- SumPDF has been refactored and behaves now as follows:
  Giving in pdfs (extended or not or mixed) *and* fracs (either length pdfs or one less) will create a
  non-extended SumPDF using the fracs. The fact that the pdfs are maybe extended is ignored.
  This will lead to highly consistent behavior.
  If the number of fracs given equals the number of pdfs, it is up to the user (currently) to take care of
  the normalization.
  *Only* if *all* pdfs are extended **and** no fracs are given, the sumpdf will be using the yields as
  normalized fracs and be extended.

- Improved graph building and ``z.function``

  * the ``z.function`` can now, as with ``tf.function``, be used either as a decorator without arguments or as a
    decorator with arguments. They are the same as in ``tf.function``, except of a few additional ones.
  * ``zfit.run.set_mode`` allows to set the policy for whether everything is run in eager mode (``graph=False``),
    everything in graph, or most of it (``graph=True``) or an optimized variant, doing graph building only with
    losses but not just models (e.g. ``pdf`` won't trigger a graph build, ``loss.value()`` will) with ``graph='auto'``.
  * The graph cache can be cleaned manually using ``zfit.run.clear_graph_cache()`` in order to prevent slowness
    in repeated tasks.

- Switch for numerical gradients has been added as well in ``zfit.run.set_mode(autograd=True/False)``.
- Resetting to the default can be done with ``zfit.run.set_mode_default()``
- Improved stability of minimizer by adding penalty (currently in ``Minuit``) as default. To have a
  better behavior with toys (e.g. never fail on NaNs but return an invalid ``FitResult``), use the
  ``DefaultToyStrategy`` in ``zfit.mnimize``.
- Exceptions are now publicly available in ``zfit.exception``
- Added nice printout for ``FitResult`` and ``FitResult.params``.
- ``get_params`` is now more meaningful, returning by default all independent parameters of the pdf, including yields.
  Arguments (``floating``, ``is_yield``) allow for more fine-grained control.

Breaking changes
------------------
- Multiple limits are now handled by a MultiSpace class. Each Space has only "one limit"
  and no complicated layout has to be remembered. If you want to have a space that is
  defined in disconnected regions, use the ``+`` operator or functionally ``zfit.dimension.add_spaces``

  To extract limits from multiple limits, ``MultiSpace`` and ``Space`` are both iterables, returning
  the containing spaces respectively itself (for the ``Space`` case).
- SumPDF changed in the behavior. Read above in the Major Features and Improvement.
- Integrals of extended PDFs are not extended anymore, but ``ext_integrate`` now returns the
  integral multiplied by the yield.

Deprecations
-------------
- ``ComposedParameter`` takes now ``params`` instead of ``dependents`` as argument, it acts now as
  the arguments to the ``value_fn``. To stay future compatible, create e.g. ``def value_fn(p1, pa2)``
  and using ``params = ['param1, param2]``, ``value_fn`` will then be called as ``value_fn(param1, parma2)``.
  ``value_fn`` without arguments will probably break in the future.
- ``FitResult.error`` has been renamed to ``errors`` to better reflect that multiple errors, the lower and
  upper are returned.


Bug fixes and small changes
---------------------------
- fix a (nasty, rounding) bug in sampling with multiple limits
- fix bug in numerical calculation
- fix bug in SimplePDF
- fix wrong caching signature may lead to graph not being rebuild
- add ``zfit.param.set_values`` method that allows to set the values of multiple
  parameters with one command. Can, as the ``set_value`` method be used with a context manager.
- wrong size of weights when applying cuts in a dataset
- ``with_coords`` did drop axes/obs
- Fix function not traced when an error was raised during first trace
- MultipleLimits support for analytic integrals
- ``zfit.param.set_values(..)`` now also can use a ``FitResult`` as ``values`` argument to set the values
  from.

Experimental
------------
- added a new error method, ``'zfit_error'`` that is equivalent to ``'minuit_minos'``, but not fully
  stable. It can be used with other minimizers as well, not only Minuit.

Requirement changes
-------------------
- remove the outdated typing module
- add tableformatter, colored, colorama for colored table printout

Thanks
------
- Johannes Lade for code review and discussions.
- Hans Dembinski for useful inputs to the uncertainties.

0.4.3 (11.3.2020)
=================


Major Features and Improvements
-------------------------------

- refactor ``hesse_np`` with covariance matrix, make it available to all minimizers

Behavioral changes
------------------


Bug fixes and small changes
---------------------------

- fix bug in ``hesse_np``


Requirement changes
-------------------


Thanks
------


0.4.2 (27.2.2020)
=================


Major Features and Improvements
-------------------------------

- Refactoring of the Constraints, dividing into ``ProbabilityConstraint`` that can be
  sampled from and more general constraints (e.g. for parameter boundaries) that
  can not be sampled from.
- Doc improvements in the constraints.
- Add ``hesse`` error method ('hesse_np') available to all minimizers (not just Minuit).


Behavioral changes
------------------
- Changed default step size to an adaptive scheme, a fraction (1e-4) of the range between the lower and upper limits.


Bug fixes and small changes
---------------------------
- Add ``use_minuit_grad`` option to Minuit optimizer to use the internal gradient, often for more stable fits
- added experimental flag ``zfit.experimental_loss_penalty_nan``, which adds a penalty to the loss in case the value is
  nan. Can help with the optimisation. Feedback welcome!

Requirement changes
-------------------


Thanks
------


0.4.1 (12.1.20)
===============

Release to keep up with TensorFlow 2.1

Major Features and Improvements
-------------------------------

- Fixed the comparison in caching the graph (implementation detail) that leads to an error.


0.4.0 (7.1.2020)
================

This release switched to TensorFlow 2.0 eager mode. In case this breaks things for you and you need **urgently**
a running version, install a version
< 0.4.1. It is highly recommended to upgrade and make the small changes required.

Please read the ``upgrade guide <docs/project/upgrade_guide.rst>`` on a more detailed explanation how to upgrade.

TensorFlow 2.0 is eager executing and uses functions to abstract the performance critical parts away.


Major Features and Improvements
-------------------------------
- Dependents (currently, and probably also in the future) need more manual tracking. This has mostly
  an effect on CompositeParameters and SimpleLoss, which now require to specify the dependents by giving
  the objects it depends (indirectly) on. For example, it is sufficient to give a ``ComplexParameter`` (which
  itself is not independent but has dependents) to a ``SimpleLoss`` as dependents (assuming the loss
  function depends on it).
- ``ComposedParameter`` does no longer allow to give a Tensor but requires a function that, when evaluated,
  returns the value. It depends on the ``dependents`` that are now required.
- Added numerical differentiation, which allows now to wrap any function with ``z.py_function`` (``zfit.z``).
  This can be switched on with ``zfit.settings.options['numerical_grad'] = True``
- Added gradient and hessian calculation options to the loss. Support numerical calculation as well.
- Add caching system for graph to prevent recursive graph building
- changed backend name to ``z`` and can be used as ``zfit.z`` or imported from it. Added:

   - ``function`` decorator that can be used to trace a function. Respects dependencies of inputs and automatically
     caches/invalidates the graph and recreates.
   - ``py_function``, same as ``tf.py_function``, but checks and may extends in the future
   - ``math`` module that contains autodiff and numerical differentiation methods, both working with tensors.

Behavioral changes
------------------
- EDM goal of the minuit minimizer has been reduced by a factor of 10 to 10E-3 in agreement with
  the goal in RooFits Minuit minimizer. This can be varied by specifying the tolerance.
- known issue: the ``projection_pdf`` has troubles with the newest TF version and may not work properly (runs out of
  memory)


Bug fixes and small changes
---------------------------

Requirement changes
-------------------
- added numdifftools (for numerical differentiation)


Thanks
------

0.3.7 (6.12.19)
================

This is a legacy release to add some fixes, next release is TF 2 eager mode only release.


Major Features and Improvements
-------------------------------
 - mostly TF 2.0 compatibility in graph mode, tests against 1.x and 2.x

Behavioral changes
------------------

Bug fixes and small changes
---------------------------
 - ``get_depentents`` returns now an OrderedSet
 - errordef is now a (hidden) attribute and can be changed
 - fix bug in polynomials


Requirement changes
-------------------
 - added ordered-set

0.3.6 (12.10.19)
================

**Special release for conda deployment and version fix (TF 2.0 is out)**

**This is the last release before breaking changes occur**


Major Features and Improvements
-------------------------------
 - added ConstantParameter and ``zfit.param`` namespace
 - Available on conda-forge

Behavioral changes
------------------
 - an implicitly created parameter with a Python numerical (e.g. when instantiating a model)
   will be converted to a ConstantParameter instead of a fixed Parameter and therefore
   cannot be set to floating later on.

Bug fixes and small changes
---------------------------
 - added native support TFP distributions for analytic sampling
 - fix Gaussian (TFP Distribution) Constraint with mixed up order of parameters

 - ``from_numpy`` automatically converts to default float regardless the original numpy dtype,
   ``dtype`` has to be used as an explicit argument


Requirement changes
-------------------
 - TensorFlow >= 1.14 is required


Thanks
------
 - Chris Burr for the conda-forge deployment


0.3.4 (30-07-19)
================

**This is the last release before breaking changes occur**

Major Features and Improvements
-------------------------------
- create ``Constraint`` class which allows for more fine grained control and information on the applied constraints.
- Added Polynomial models
- Improved and fixed sampling (can still be slightly biased)

Behavioral changes
------------------
None

Bug fixes and small changes
---------------------------

- fixed various small bugs

Thanks
------
for the contribution of the Constraints to Matthieu Marinangeli <matthieu.marinangeli@cern.ch>



0.3.3 (15-05-19)
================

Fixed Partial numeric integration

Bugfixes mostly, a few major fixes. Partial numeric integration works now.

Bugfixes
 - ``data_range`` cuts are now applied correctly, also in several dimensions when a subset is selected
   (which happens internally of some Functors, e.g. ProductPDF). Before, only the selected obs was respected for cuts.
 - parital integration had a wrong take on checking limits (now uses supports).


0.3.2 (01-05-19)
================

With 0.3.2, bugfixes and three changes in the API/behavior

Breaking changes
----------------
 - tfp distributions wrapping is now different with ``dist_kwargs`` allowing for non-Parameter arguments (like other dists)
 - sampling allows now for importance sampling (sampler in Model specified differently)
 - ``model.sample`` now also returns a tensor, being consistent with ``pdf`` and ``integrate``

Bugfixes
--------
 - shape handling of tfp dists was "wrong" (though not producing wrong results!), fixed. TFP distributions now get a tensor with shape (nevents, nobs) instead of a list of tensors with (nevents,)

Improvements
------------
 - refactor the sampling for more flexibility and performance (less graph constructed)
 - allow to use more sophisticated importance sampling (e.g. phasespace)
 - on-the-fly normalization (experimentally) implemented with correct gradient



0.3.1 (30-04-19)
================


Minor improvements and bugfixes including:

- improved importance sampling allowing to preinstantiate objects before it's called inside the while loop
- fixing a problem with ``ztf.sqrt``



0.3.0 (2019-03-20)
==================


Beta stage and first pip release


0.0.1 (2018-03-22)
==================


- First creation of the package.
