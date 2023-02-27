#  Copyright (c) 2023 zfit

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

    def _unnormalized_pdf(self, x):
        energy, momentum = x.unstack_x()
        param1 = self.params["super_param"]
        param2 = self.params["param2"]
        param3 = self.params["param3"]

        # just a fantasy function
        probs = (
            param1 * tf.cos(energy**2) + tf.math.log(param2 * momentum**2) + param3
        )
        return probs


# add an analytic integral


# define the integral function
def integral_full(limits, norm_range, params, model):
    (
        lower,
        upper,
    ) = limits.rect_limits  # for a more detailed guide, see the space.py example
    param1 = params["super_param"]
    param2 = params["param2"]
    param3 = params["param3"]

    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)

    # calculate the integral here, dummy integral, wrong!
    integral = param1 * param2 * param3 + z.reduce_sum([lower, upper])
    return integral


# define the space over which it is defined. Here, we use the axes
lower_full = (-10, zfit.Space.ANY_LOWER)
upper_full = (10, zfit.Space.ANY_UPPER)
integral_full_limits = zfit.Space(axes=(0, 1), limits=(lower_full, upper_full))

CustomPDF2D.register_analytic_integral(func=integral_full, limits=integral_full_limits)


# define the partial integral function
def integral_axis1(x, limits, norm_range, params, model):
    data_0 = x.unstack_x()  # data from axis 0

    param1 = params["super_param"]
    param2 = params["param2"]
    param3 = params["param3"]

    lower, upper = limits.limit1d  # for a more detailed guide, see the space.py example
    lower = z.convert_to_tensor(lower)  # the limits are now 1-D, for axis 1
    upper = z.convert_to_tensor(upper)

    # calculate the integral here, dummy integral
    integral = data_0**2 * param1 * param2 * param3 + z.reduce_sum([lower, upper])
    # notice that the returned shape will be in the same as data_0, e.g. the number of events given in x
    return integral


# define the space over which it is defined. Here, we use the axes
lower_axis1 = ((zfit.Space.ANY_LOWER,),)
upper_axis1 = ((zfit.Space.ANY_UPPER,),)
integral_axis1_limits = zfit.Space(
    axes=(1,),  # axes one corresponds to the second obs, here obs2
    limits=(lower_axis1, upper_axis1),
)

CustomPDF2D.register_analytic_integral(
    func=integral_axis1, limits=integral_axis1_limits
)

if __name__ == "__main__":
    import numpy as np

    obs = zfit.Space("obs1", (-10, 10)) * zfit.Space("obs2", (-3, 5))
    pdf = CustomPDF2D(1, 2, 3, obs=obs)
    sample = pdf.sample(n=1000)
    pdf.pdf([[2.0, 2.5], [5.4, 3.2]])
    x_part = zfit.Data.from_numpy(array=np.array([2.1, 2.2, 3.2]), obs="obs1")

    # integrate over obs2 with limits 1, 2 for the `x_part`. This will use the analytic integral above
    pdf.partial_integrate(x=x_part, limits=zfit.Space("obs2", (1, 2)))
    # we can explicitly call the analytic integral. Without registering it (e.g. comment the line with the `register`
    # and run again), it will raise an error
    pdf.partial_analytic_integrate(x=x_part, limits=zfit.Space("obs2", (1, 2)))
