*********
Changelog
*********

Develop
=======



Major Features and Improvements
-------------------------------


Behavioral changes
------------------


Bug fixes and small changes
---------------------------

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

Please read the `upgrade guide <docs/project/upgrade_guide.rst>` on a more detailed explanation how to upgrade.

TensorFlow 2.0 is eager executing and uses functions to abstract the performance critical parts away.


Major Features and Improvements
-------------------------------
 - Dependents (currently, and probably also in the future) need more manual tracking. This has mostly
   an effect on CompositeParameters and SimpleLoss, which now require to specify the dependents by giving
   the objects it depends (indirectly) on. For example, it is sufficient to give a `ComplexParameter` (which
   itself is not independent but has dependents) to a `SimpleLoss` as dependents (assuming the loss
   function depends on it).
 - `ComposedParameter` does no longer allow to give a Tensor but requires a function that, when evaluated,
   returns the value. It depends on the `dependents` that are now required.
 - Added numerical differentiation, which allows now to wrap any function with `z.py_function` (`zfit.z`).
   This can be switched on with `zfit.settings.options['numerical_grad'] = True`
 - Added gradient and hessian calculation options to the loss. Support numerical calculation as well.
 - Add caching system for graph to prevent recursive graph building
 - changed backend name to `z` and can be used as `zfit.z` or imported from it. Added:

    - `function` decorator that can be used to trace a function. Respects dependencies of inputs and automatically
      caches/invalidates the graph and recreates.
    - `py_function`, same as `tf.py_function`, but checks and may extends in the future
    - `math` module that contains autodiff and numerical differentiation methods, both working with tensors.


Behavioral changes
------------------
 - EDM goal of the minuit minimizer has been reduced by a factor of 10 to 10E-3 in agreement with
   the goal in RooFits Minuit minimizer. This can be varied by specifying the tolerance.
 - known issue: the `projection_pdf` has troubles with the newest TF version and may not work properly (runs out of
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
 - `get_depentents` returns now an OrderedSet
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
 - added ConstantParameter and `zfit.param` namespace
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

 - `from_numpy` automatically converts to default float regardless the original numpy dtype,
   `dtype` has to be used as an explicit argument


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
- create `Constraint` class which allows for more fine grained control and information on the applied constraints.
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
 - `model.sample` now also returns a tensor, being consistent with `pdf` and `integrate`

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
- fixing a problem with `ztf.sqrt`



0.3.0 (2019-03-20)
==================


Beta stage and first pip release


0.0.1 (2018-03-22)
==================


* First creation of the package.
