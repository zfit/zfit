#  Copyright (c) 2019 zfit
#
# import zfit
# import zfit.models.dist_tfp
#
#
# def test_simple_kde():
#     h = zfit.Parameter("h", 1.)
#     data = np.random.normal(size=100)
#     kde = zfit.models.dist_tfp.KernelDensity(loc=data, scale=h, obs=zfit.Space("obs1", limits=(-5, 5)))
#
#     integral = kde.integrate(limits=(-5, 5))
#     assert zfit.run(integral) == pytest.approx(1., rel=0.01)
