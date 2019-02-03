import numpy as np
import tensorflow as tf

import zfit
from zfit import ztf

from zfit.core.basepdf import BasePDF
from zfit.util import ztyping
from zfit.core.limits import Space, ANY_LOWER, ANY_UPPER
from zfit.core.loss import UnbinnedNLL
from zfit.minimizers.minimizer_minuit import MinuitMinimizer

# Hack
import matplotlib

matplotlib.use('TkAgg')
# Hack End
import matplotlib.pyplot as plt

import os

from timeit import default_timer


def memory_usage():
    """Get memory usage of current process in MiB.

    Tries to use :mod:`psutil`, if possible, otherwise fallback to calling
    ``ps`` directly.

    Return:
        float: Memory usage of the current process.

    """
    pid = os.getpid()
    try:
        import psutil
        process = psutil.Process(pid)
        mem = process.memory_info()[0] / float(2 ** 20)
    except ImportError:
        import subprocess
        out = subprocess.Popen(['ps', 'v', '-p', str(pid)],
                               stdout=subprocess.PIPE).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024
    return mem


# pylint: disable=too-few-public-methods
class Timer(object):
    """Time the code placed inside its context.

    Taken from http://coreygoldberg.blogspot.ch/2012/06/python-timer-class-context-manager-for.html

    Attributes:
        verbose (bool): Print the elapsed time at context exit?
        start (float): Start time in seconds since Epoch Time. Value set
            to 0 if not run.
        elapsed (float): Elapsed seconds in the timer. Value set to
            0 if not run.

    Arguments:
        verbose (bool, optional): Print the elapsed time at
            context exit? Defaults to False.

    """

    def __init__(self, verbose=False):
        """Initialize the timer."""
        self.verbose = verbose
        self._timer = default_timer
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = self._timer()
        return self

    def __exit__(self, *args):
        self.elapsed = self._timer() - self.start
        if self.verbose:
            print('Elapsed time: {} ms'.format(self.elapsed * 1000.0))


# EOF


class AngularPDF(BasePDF):
    """Angular distribution for Lb->Lz gamma."""

    def __init__(self, a, obs: ztyping.ObsTypeInput, name: str = "Angular",
                 **kwargs):
        """Angular distribution in the form 1-a*cos(theta).

        The function is normalized over a finite range and therefore a PDF.

        Args:
            a (zfit.Parameter): Accessed as "a".
            obs (Space): The Space the pdf is defined in.
            name (str): Name of the pdf.
            dtype (DType):

        """
        parameters = {'a': a}
        super().__init__(obs, name=name, parameters=parameters, **kwargs)

    def _unnormalized_pdf(self, costheta):
        a = self.parameters['a']
        costheta = ztf.unstack_x(costheta)
        return 1 - a * costheta


def _ang_integral_from_any_to_any(limits, params):
    """Angular distribution integral."""
    a = params['a']

    def raw_integral(x):
        return x - a / 2 * tf.pow(x, 2)

    (lower,), (upper,) = limits.limits
    if lower[0] == - upper[0] == np.inf:
        raise NotImplementedError
    lower_int = raw_integral(x=ztf.constant(lower))
    upper_int = raw_integral(x=ztf.constant(upper))
    return upper_int - lower_int


limits = Space.from_axes(axes=0, limits=(ANY_LOWER, ANY_UPPER))
AngularPDF.register_analytic_integral(func=_ang_integral_from_any_to_any, limits=limits)


# Plot to test
def plot_pdf(pdf, norm=1, n_points=1000):
    obs = pdf.norm_range
    lower, upper = obs.limits
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=n_points)
    y_plot = zfit.run(pdf.pdf(x_plot, norm_range=obs))
    plt.plot(x_plot, y_plot * norm)


def plot_data(data_tensor, n_bins=50):
    data_np = zfit.run(data_tensor)
    counts, bin_edges = np.histogram(data_np, n_bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    err = np.sqrt(counts)
    plt.errorbar(bin_centres, counts, yerr=err, fmt='o')
    return data_np.shape


def plot_nll_scan(nll, param: zfit.Parameter, n_points=1000):
    lower, upper = param.lower_limit, param.upper_limit
    x_plot = np.linspace(zfit.run(lower), zfit.run(upper), num=n_points)
    y_plot = []
    for x in x_plot:
        param.load(x)
        y_plot.append(zfit.run(nll.value()))
    plt.title("NLL scan of {}".format(param.name))
    plt.plot(x_plot, y_plot)


N_BINS = 20
N_EVENTS = 150
N_TOYS = 1000
DO_SIMUL = False

if __name__ == "__main__":
    # Observables
    cos_gamma = zfit.Space("cos_gamma", limits=(-1, 1))
    cos_theta = zfit.Space("cos_theta", limits=(-1, 1))
    # Parameters
    alpha_gamma = zfit.Parameter("alpha_gamma", 1, -2, 2)
    lb_pol = zfit.Parameter("lb_pol", 0.06, 0, 1)
    alpha_p = zfit.Parameter("alpha_p", 0.642, 0, 1)
    # Constraints
    lb_pol_constraint = zfit.constraint.nll_gaussian(params=lb_pol,
                                                     mu=0.06,
                                                     sigma=0.08)
    alpha_p_constraint = zfit.constraint.nll_gaussian(params=alpha_p,
                                                      mu=0.642,
                                                      sigma=0.013)
    # PDFs
    cos_gamma_pdf = AngularPDF(alpha_gamma * lb_pol, obs=cos_gamma)
    cos_theta_pdf = AngularPDF(alpha_gamma * alpha_p, obs=cos_theta)
    # plot_pdf(cos_gamma_pdf)
    # plot_pdf(cos_theta_pdf)
    # plt.show()

    cos_gamma_data = cos_gamma_pdf.sample(n=N_EVENTS, limits=cos_gamma)
    cos_theta_data = cos_theta_pdf.sample(n=N_EVENTS, limits=cos_theta)
    cos_gamma_data = tf.Variable(initial_value=cos_gamma_data, use_resource=True)
    cos_theta_data = tf.Variable(initial_value=cos_theta_data, use_resource=True)
    # Losses
    cos_gamma_nll = UnbinnedNLL(model=cos_gamma_pdf,
                                data=cos_gamma_data,
                                fit_range=(-1, 1),
                                constraints=lb_pol_constraint)
    cos_theta_nll = UnbinnedNLL(model=cos_theta_pdf,
                                data=cos_theta_data,
                                fit_range=(-1, 1),
                                constraints=alpha_p_constraint)
    full_nll = cos_gamma_nll + cos_theta_nll

    # Load and instantiate the Minuit minimiser
    minimizer = MinuitMinimizer()

    # Minimize cos theta
    alpha_gamma.randomize()

    alpha_vals_costheta = []
    status_costheta = []
    alpha_vals_full = []
    mems = []
    times = []
    failed_fits = 0
    have_nan = 0

    graph = tf.get_default_graph()

    while len(alpha_vals_costheta) <= N_TOYS:
        with Timer() as t:
            mems.append(memory_usage())
            # Reset variables
            alpha_gamma.load(1)
            lb_pol.load(0.06)
            alpha_p.load(0.642)
            # Let's generate some data
            # Hack
            zfit.run([cos_gamma_data.initializer, cos_theta_data.initializer])  # this actually runs the sampling
            # Hack end
            # Randomize variables
            alpha_gamma.randomize()
            lb_pol.randomize()
            alpha_p.randomize()
            # Minimize costheta
            minimum = minimizer.minimize(loss=cos_theta_nll, params=[alpha_gamma, alpha_p])
            # Plot result
            # plot_pdf(cos_theta_pdf, plot_data(cos_theta_data, N_BINS)[0]/N_BINS*2)
            # And NLL profile
            # plt.figure()
            # plot_nll_scan(cos_theta_nll, param=alpha_gamma)
            if minimum.converged and not np.isnan((minimum.params[alpha_gamma]['value'])):
                alpha_vals_costheta.append(minimum.params[alpha_gamma]['value'])
                status_costheta.append(minimum.status)
            else:
                failed_fits += 1
                if np.isnan((minimum.params[alpha_gamma]['value'])):
                    have_nan += 1
            if DO_SIMUL:
                # Minimize simultaneously
                alpha_gamma.randomize()
                lb_pol.randomize()
                alpha_p.randomize()
                minimum = minimizer.minimize(loss=full_nll, params=[alpha_gamma, alpha_p, lb_pol])
                alpha_vals_full.append(minimum.params[alpha_gamma]['value'])
            graph.finalize()

        times.append(t.elapsed * 1000.0)

    print("Failed fits: {}/{} ({} with NaN)".format(failed_fits,
                                                    failed_fits + N_TOYS,
                                                    have_nan))
    print("Single fit sensitivity: {}".format(np.array(alpha_vals_costheta).std()))
    if DO_SIMUL:
        print("Simultaneous fit sensitivity: {}".format(np.array(alpha_vals_full).std()))
    print("Time per toy: {:.0f} ms".format(np.array(times).mean()))
    print("Memory increase per toy: {} MB".format((mems[-1] - mems[0]) / N_TOYS))
    plt.hist(np.array(alpha_vals_costheta), bins=10)
    plt.savefig("costheta.png")
    plt.clf()
    plt.plot(np.array(range(times)), np.array(times), label="Time per plot")
    plt.savefig("time.png")
    plt.clf()
    plt.plot(np.array(range(mems)), np.array(mems), label="Memory evolution")
    plt.savefig("mems.png")
