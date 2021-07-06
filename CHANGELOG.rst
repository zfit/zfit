*********
Changelog
*********

.. _newest-changelog:

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
- Scipy minimizers with hessian arguments use now `BFGS` as default


Requirement changes
-------------------

- remove Python 3.6 support



0.6.6 (12.05.2021)
==================

Update ipyopt requirement < 0.12 to allow numpy compatible with TensorFlow

0.6.5 (04.05.2021)
==================

- hotfix for wrong argument in exponential PDF
- removed requirement ipyopt, can be installed with `pip install zfit[ipyopt]`
  or by manually installing `pip install ipyopt`



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
  - a minimization can be 'continued' by passing `init` to `minimize`
  - more streamlined arguments for minimizers, harmonized names and behavior.
  - Adding a flexible criterion (currently EDM) that will terminate the minimization.
  - Making the minimizer fully stateless.
  - Moving the loss evaluation and strategy into a LossEval that simplifies the handling of printing and NaNs.
  - Callbacks are added to the strategy.

- Major overhaul of the ``FitResult``, including:

  - improved `zfit_error` (equivalent of `MINOS`)
  - `minuit_hesse` and `minuit_minos` are now available with all minimizers as well thanks to an great
    improvement in iminuit.
  - Added an `approx` hesse that returns the approximate hessian (if available, otherwise empty)

- upgrade to iminuit v2 changes the way it works and also the Minuit minimizer in zfit,
  including a new step size heuristic.
  Possible problems can be caused by iminuit itself, please report
  in case your fits don't converge anymore.
- improved ``compute_errors`` in speed by caching values and the reliability
  by making the solution unique.
- increased stability for large datasets with a constant subtraction in the NLL

Breaking changes
------------------
- NLL (and extended) subtracts now by default a constant value. This can be changed with a new `options` argument.
  COMPARISON OF DIFFEREN NLLs (their absolute values) fails now! (flag can be deactivated)
- BFGS (from TensorFlow Probability) has been removed as it is not working properly. There are many alternatives
  such as ScipyLBFGSV1 or NLoptLBFGSV1
- Scipy (the minimizer) has been removed. Use specialized `Scipy*` minimizers instead.
- Creating a ``zfit.Parameter``, usign ``set_value`` or ``set_values`` now raises a ``ValueError``
  if the value is outside the limits. Use ``assign`` to suppress it.

Depreceations
-------------
- strategy to minimizer should now be a class, not an instance anymore.

Bug fixes and small changes
---------------------------
- ``zfit_error`` moved only one parameter to the correct initial position. Speedup and more reliable.
- FFTconv was shifted if the kernel limits were not symetrical, now properly taken into account.
- circumvent overflow error in sampling
- shuffle samples from sum pdfs to ensure uniformity and remove conv sampling bias
- `create_sampler` now samples immediately to allow for precompile, a new hook that will allow objects to optimize
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
- new Poisson PDF

Breaking changes
------------------

Depreceations
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
- tensorflow_probability==0.11
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

Depreceations
-------------


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

  The limits returned are now by default numpy arrays with the shape (1, n_obs). This corresponds well
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
- added a new error method, 'zfit_error' that is equivalent to 'minuit_minos', but not fully
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
 - data_range cuts are now applied correctly, also in several dimensions when a subset is selected
   (which happens internally of some Functors, e.g. ProductPDF). Before, only the selected obs was respected for cuts.
 - parital integration had a wrong take on checking limits (now uses supports).


0.3.2 (01-05-19)
================

With 0.3.2, bugfixes and three changes in the API/behavior

Breaking changes
----------------
 - tfp distributions wrapping is now different with dist_kwargs allowing for non-Parameter arguments (like other dists)
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
