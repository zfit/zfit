#  Copyright (c) 2024 zfit
from __future__ import annotations

import tensorflow as tf

import zfit
from zfit import z


class CustomPDF2D(zfit.pdf.BasePDF):
    """My custom, 2 dimensional pdf where the axes are: Energy, Momentum."""

    def __init__(
        self,
        param1,
        param2,
        param3,
        obs,
        name="CustomPDF",
    ):
        # we can now do complicated stuff here if needed
        # only thing: we have to specify explicitly here what is which parameter
        params = {
            "super_param": param1,  # we can change/compose etc parameters
            "param2": param2,
            "param3": param3,
        }
        super().__init__(obs, params, name=name)

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):
        energy = x[0]
        momentum = x[1]
        param1 = params["super_param"]
        param2 = params["param2"]
        param3 = params["param3"]

        # just a fantasy function
        return param1 * tf.cos(energy**2) + tf.math.log(param2 * momentum**2) + param3


# add an analytic integral


# define the integral function
def integral_full(limits, norm, params, model):
    del norm, model  # not used here
    lower, upper = limits.v1.limits
    param1 = params["super_param"]
    param2 = params["param2"]
    param3 = params["param3"]

    # calculate the integral here, dummy integral, wrong!
    return param1 * param2 * param3 + z.reduce_sum([lower, upper])


# define the space over which it is defined. Here, we use the axes
integlimits_axis0 = zfit.Space(axes=0, lower=-10, upper=10)
integlimits_axis1 = zfit.Space(axes=1, lower=zfit.Space.ANY_LOWER, upper=zfit.Space.ANY_UPPER)

integral_full_limits = integlimits_axis0 * integlimits_axis1  # creates 2D space
CustomPDF2D.register_analytic_integral(func=integral_full, limits=integral_full_limits)


# define the partial integral function
def integral_axis1(x, limits, norm, params, model):
    del norm, model  # not used here

    data_0 = x[0]  # data from axis 0

    param1 = params["super_param"]
    param2 = params["param2"]
    param3 = params["param3"]

    lower, upper = limits.limit1d  # for a more detailed guide, see the space.py example
    lower = z.convert_to_tensor(lower)  # the limits are now 1-D, for axis 1
    upper = z.convert_to_tensor(upper)

    # calculate the integral here, dummy integral
    return data_0**2 * param1 * param2 * param3 + z.reduce_sum([lower, upper])
    # notice that the returned shape will be in the same as data_0, e.g. the number of events given in x


# define the space over which it is defined. Here, we use the axes
integral_axis1_limits = zfit.Space(
    axes=(1,),  # axes one corresponds to the second obs, here obs2
    lower=zfit.Space.ANY_LOWER,
    upper=zfit.Space.ANY_UPPER,
)

CustomPDF2D.register_analytic_integral(func=integral_axis1, limits=integral_axis1_limits)

if __name__ == "__main__":
    import numpy as np

    obs = zfit.Space("obs1", -10, 10) * zfit.Space("obs2", -3, 5)
    pdf = CustomPDF2D(
        param1=1, param2=2, param3=3, obs=obs
    )  # if a Python number is passed, it will be regarded as a constant
    sample = pdf.sample(n=1000)
    pdf.pdf([[2.0, 2.5], [5.4, 3.2]])
    # x_part = zfit.Data(np.array([2.1, 2.2, 3.2]), obs="obs1")
    x_part = np.array([2.1, 2.2, 3.2])
    # integrate over obs2 with limits 1, 2 for the `x_part`. This will use the analytic integral above
    pdf.partial_integrate(x=x_part, limits=zfit.Space("obs2", 1, 2))
    # we can explicitly call the analytic integral. Without registering it (e.g. comment the line with the `register`
    # and run again), it will raise an error
    pdf.partial_analytic_integrate(x=x_part, limits=zfit.Space("obs2", 1, 2))
