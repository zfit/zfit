*********
Changelog
*********



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

Bug fixes and other changes
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
