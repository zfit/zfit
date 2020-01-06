#  Copyright (c) 2019 zfit

import tensorflow as tf
import zfit
from zfit import z


class CustomPDF2D(zfit.pdf.BasePDF):
    """My custom, 2 dimensional pdf. The axes are: Energy, Momentum.

    """

    def __init__(self, param1, param2, param3, obs, name="CustomPDF", ):
        # we can now do complicated stuff here if needed
        # only thing: we have to specify explicitly here what is which parameter
        params = {'super_param': param1,  # we can change/compose etc parameters
                  'param2': param2, 'param3': param3}
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x):
        energy, momentum = x.unstack_x()
        param1 = self.params['super_param']
        param2 = self.params['param2']
        param3 = self.params['param3']

        # just a fantasy function
        probs = param1 * tf.cos(energy ** 2) + tf.math.log(param2 * momentum) + param3
        return probs


# add an analytic integral

# define the integral function
def integral_full(x, limits, norm_range, params, model):
    lower, upper = limits.limit1d
    param1 = params['super_param']
    param2 = params['param2']
    param3 = params['param3']

    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)

    # calculate the integral here, dummy integral
    integral = param1 * param2 * param3 + z.reduce_sum([lower, upper])
    return integral


# define the space over which it is defined. Here, we use the axes
lower_full = ((-10, zfit.Space.ANY_LOWER),)
upper_full = ((10, zfit.Space.ANY_UPPER),)
integral_full_limits = zfit.Space.from_axes(axes=(0, 1),
                                            limits=(lower_full, upper_full))

CustomPDF2D.register_analytic_integral(func=integral_full,
                                       limits=integral_full_limits)


# define the partial integral function
def integral_axis1(x, limits, norm_range, params, model):
    data_0 = x.unstack_x()  # data from axis 0

    param1 = params['super_param']
    param2 = params['param2']
    param3 = params['param3']

    lower, upper = limits.limit1d
    lower = z.convert_to_tensor(lower)  # the limits are now 1-D, for axis 1
    upper = z.convert_to_tensor(upper)

    # calculate the integral here, dummy integral
    integral = data_0 * param1 * param2 * param3 + z.reduce_sum([lower, upper])
    return integral


# define the space over which it is defined. Here, we use the axes
lower_axis1 = ((-5,),)
upper_axis1 = ((zfit.Space.ANY_UPPER,),)
integral_axis1_limits = zfit.Space.from_axes(axes=(1,),
                                             limits=(lower_axis1, upper_axis1))

CustomPDF2D.register_analytic_integral(func=integral_axis1,
                                       limits=integral_axis1_limits)
