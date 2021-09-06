Kerned Density Estimation
""""""""""""""""""""""""""""

An introduction to Kernel Density Estimations and explanations to all methods implemented in zfit can be found in
`Performance of univariate kernel density estimation methods in TensorFlow <https://astroviking.github.io/ba-thesis/>`_
by Marc Steiner.

Estimating the density of a population can be done in a non-parametric manner. The simplest form is to create a
density histogram, which is notably not so precise.

A more sophisticated non-parametric method is the kernel density estimation (KDE), which can be looked at as a sort of
generalized histogram. In a kernel density estimation each data point is substituted with a so called kernel function
that specifies how much it influences its neighboring regions. This kernel functions can then be summed up to get an
estimate of the probability density distribution, quite similarly as summing up data points inside bins.

However, since
the kernel functions are centered on the data points directly, KDE circumvents the problem of arbitrary bin positioning.
Since KDE still depends on kernel bandwidth (a measure of the spread of the kernel function) instead of bin width,
one might argue that this is not a major improvement. Upon closer inspection, one finds that the underlying PDF
does depend less strongly on the kernel bandwidth than histograms do on bin width and it is much easier to specify
rules for an approximately optimal kernel bandwidth than it is to do so for bin width.

Exact KDEs
''''''''''

**Summary**
*An exact KDE calculates the true sum of the kernels around the input points without approximating the
dataset. The computational complexity -- especially the peak memory consumption -- is proportional to the sample size.
Therefore, this kind of PDF is better used for smaller datasets and a Grid KDE is preferred for larger data.*

Given a set of :math:`$n$` sample points :math:`$x_k$` (:math:`$k = 1,\cdots,n$`), an exact kernel density estimation
`$\widehat{f}_h(x)$` can be calculated as

.. math::
    \widehat{f}_h(x) = \frac{1}{nh} \sum_{k=1}^n K\Big(\frac{x-x_k}{h}\Big)

where :math:`$K(x)$` is called the kernel function, :math:`$h$` is the bandwidth of the kernel and :math:`$x$` is the
value for which the estimate is calculated. The kernel function defines the shape and size of influence of a single
data point over the estimation, whereas the bandwidth defines the range of influence. Most typically a simple
Gaussian distribution (:math:`$K(x) :=\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}$`) is used as kernel function.
The larger the bandwidth parameter $h$ the larger is the range of influence of
a single data point on the estimated distribution.

The computational complexity of the exact KDE above is given by :math:`$\mathcal{O}(nm)$` where :math:`$n$`
is the number of sample points to estimate from and :math:`$m$` is the number of evaluation points
(the points where you want to calculate the estimate).

There exist several approximative methods to decrease this complexity and therefore decrease the runtime as well.

In zfit, the exact KDE :py:class:~`zfit.pdf.ExactKDE1Dim` takes an arbitrary kernel, which is a
TensorFlow-Probability distribution.

.. code-block::

  kde = zfit.pdf.ExactKDE1Dim(obs, data, bandwidth, kernel, padding, weights, name)


Grid KDEs
'''''''''

The most straightforward way to decrease the computational complexity is by limiting the number of sample points.
This can be done by a binning routine, where the values at a smaller number of regular grid points are estimated
from the original larger number of sample points.
Given a set of sample points :math:`$X = \{x_0, x_1, ..., x_k, ..., x_{n-1}, x_n\}$` with weights $w_k$ and a set of
equally spaced grid points :math:`$G = \{g_0, g_1, ..., g_l, ..., g_{n-1}, g_N\}$` where :math:`$N < n$`
we can assign an estimate
(or a count) :math:`$c_l$` to each grid point :math:`$g_l$` and use the newly found :math:`$g_l$` to calculate
the kernel density estimation instead.

.. math::
    \widehat{f}_h(x) = \frac{1}{nh} \sum_{l=1}^N c_l \cdot K\Big(\frac{x-g_l}{h}\Big)

This lowers the computational complexity down to :math:`$\mathcal{O}(N \cdot m)$`.
Depending on the number of grid points :math:`$N$` there is tradeoff between accuracy and speed.
However as we will see in the comparison chapter later as well, even for ten million sample points, a grid of size
:math:`$1024$` is enough to capture the true density with high accuracy. As described in the extensive overview
by Artur Gramacki[@gramacki2018fft], simple binning or linear binning can be used, although the last is often
preferred since it is more accurate and the difference in computational complexity is negligible.
