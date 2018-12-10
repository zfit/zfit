import array

import numpy as np
import tensorflow as tf

try:
    from ROOT import TVirtualFitter, TNtuple, TH1, TH2, TH3
except ModuleNotFoundError:
    print("Legacy mode active, ROOT not loaded")

from zfit.core.math import interpolate
from zfit.settings import types as ztypes


class RootHistShape(object):
    """
      Class that creates a TensorFlow graph from a bilinear interpolation of
      ROOT TH{1,2,3} histogram. Useful for e.g. efficiency and background shapes
      stored in histograms.
    """

    def __init__(self, hist):
        """
          Constructor.
            hist - ROOT THn object
        """
        if isinstance(hist, TH1):
            nx = hist.GetNbinsX()
            data = np.zeros(nx, dtype=np.dtype('d'))
            self.limits = [
                tf.constant([hist.GetXaxis().GetBinCenter(1)], dtype=ztypes.float),
                tf.constant([hist.GetXaxis().GetBinCenter(nx)], dtype=ztypes.float),
                ]
            for x in range(nx):
                data[x] = hist.GetBinContent(x + 1)
            self.ns = tf.constant([nx - 1], dtype=ztypes.float)

        if isinstance(hist, TH2):
            nx = hist.GetNbinsX()
            ny = hist.GetNbinsY()
            data = np.zeros((nx, ny), dtype=np.dtype('d'))
            self.limits = [
                tf.constant([hist.GetXaxis().GetBinCenter(1), hist.GetYaxis().GetBinCenter(1)],
                            dtype=ztypes.float),
                tf.constant([hist.GetXaxis().GetBinCenter(nx), hist.GetYaxis().GetBinCenter(ny)],
                            dtype=ztypes.float),
                ]
            for x in range(nx):
                for y in range(ny):
                    data[x][y] = hist.GetBinContent(x + 1, y + 1)
            self.ns = tf.constant([nx - 1, ny - 1], dtype=ztypes.float)

        if isinstance(hist, TH3):
            nx = hist.GetNbinsX()
            ny = hist.GetNbinsY()
            nz = hist.GetNbinsZ()
            data = np.zeros((nx, ny, nz), dtype=np.dtype('d'))
            self.limits = [
                tf.constant([hist.GetXaxis().GetBinCenter(1), hist.GetYaxis().GetBinCenter(1),
                             hist.GetZaxis().GetBinCenter(1)], dtype=ztypes.float),
                tf.constant([hist.GetXaxis().GetBinCenter(nx), hist.GetYaxis().GetBinCenter(ny),
                             hist.GetZaxis().GetBinCenter(nz)], dtype=ztypes.float),
                ]
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        data[x][y][z] = hist.GetBinContent(x + 1, y + 1, z + 1)
            self.ns = tf.constant([nx - 1, ny - 1, nz - 1], dtype=ztypes.float)

        self.array = tf.constant(data, dtype=ztypes.float)

    def shape(self, x):
        """
          Method that returns a TF graph with the interpolation result for for a set of N
          M-dimensional points
            x - TF tensor of shape (N, M)
        """
        c = (x - self.limits[0]) / (self.limits[1] - self.limits[0]) * self.ns
        return interpolate(self.array, c)


def fill_NTuple(tupname, data, names):
    """
      Create and fill ROOT NTuple with the data sample.
        tupname : name of the NTuple
        data : data sample
        names : names of the NTuple variables
    """
    variables = ""
    for n in names:
        variables += "{}:".format(n)
    variables = variables[:-1]
    values = len(names) * [0.]
    avalues = array.array('f', values)
    nt = TNtuple(tupname, "", variables)
    for d in data:
        for i in range(len(names)):
            avalues[i] = d[i]
        nt.Fill(avalues)
    nt.Write()


def read_NTuple(ntuple, variables):
    """
      Return a numpy tuple_array with the value from TNtuple.
        ntuple : input TNtuple
        variables : list of ntuple variables to read
    """
    data = []  # TODO: smell, unused?
    code_list = []
    for v in variables:
        code_list += [compile("i.{}".format(v), '<string>', 'value')]
    nentries = ntuple.GetEntries()
    nvars = len(variables)
    tuple_array = np.zeros((nentries, nvars))
    for n, i in enumerate(ntuple):
        for m, v in enumerate(code_list):
            tuple_array[n][m] = eval(v)
        if n % 100000 == 0:
            print(n, "/", nentries)
    return tuple_array


def run_minuit(sess, nll, feed_dict=None, float_tfpars=None, call_limit=50000, use_gradient=True,
               gradient=None, printout=50, tmp_file="tmp_result.txt",
               run_hesse=False, run_minos=False,
               options=None, run_metadata=None):
    """
      Perform MINUIT minimisation of the negative likelihood.
        sess         : TF session
        nll          : graph for negitive likelihood to be minimised
        feed_dict    :
        float_tfpars : list of floating parameter to be used in the fit
        call_limit   : call limit for MINUIT
        gradient_par     : external gradient_par graph. If None and use_gradient is not False, will be
                       calculated internally
        use_gradient  : flag to control the use of analytic gradient_par while fitting:
                       None or False   : gradient_par is not used
                       True or "CHECK" : analytic gradient_par will be checked with finite elements,
                                        and will be used is they match
                       "FORCE"         : analytic gradient_par will be used regardless.
        printout     : Printout frequency
        tmp_file      : Name of the file with temporary results (updated every time printout is
        called)
        run_hesse     ; Run HESSE after minimisation
        run_minos     : Run MINOS after minimisation
        options      : additional options to pass to TF session run
        run_metadata : metadata to pass to TF session run
    """

    global cacheable_tensors

    if float_tfpars is None:
        tfpars = tf.trainable_variables()  # Create TF variables
        float_tfpars = [p for p in tfpars if p.floating()]  # List of floating parameters

    if use_gradient and gradient is None:
        gradient = tf.gradients(nll, float_tfpars)  # Get analytic gradient_par

    cached_data = {}

    fetch_list = []
    for i in cacheable_tensors:
        if i not in cached_data:
            fetch_list += [i]
    if feed_dict:
        feeds = dict(feed_dict)
    else:
        feeds = None
    for i in cacheable_tensors:
        if i in cached_data:
            feeds[i] = cached_data[i]

    fetch_data = sess.run(fetch_list, feed_dict=feeds)  # Calculate log likelihood

    for i, d in zip(fetch_list, fetch_data):
        cached_data[i] = d

    if feed_dict:
        feeds = dict(feed_dict)
    else:
        feeds = None
    for i in cacheable_tensors:
        if i in cached_data:
            feeds[i] = cached_data[i]

    def fcn(npar, gin, f, par, istatus):  # MINUIT fit function
        for i in range(len(float_tfpars)):
            float_tfpars[i].update(sess, par[i])

        f[0] = sess.run(nll, feed_dict=feeds, options=options,
                        run_metadata=run_metadata)  # Calculate log likelihood

        if istatus == 2:  # If gradient_par calculation is needed
            dnll = sess.run(gradient, feed_dict=feeds, options=options,
                            run_metadata=run_metadata)  # Calculate analytic gradient_par
            for j in range(len(float_tfpars)):
                gin[j] = dnll[j]  # Pass gradient_par to MINUIT
        fcn.n += 1
        if fcn.n % printout == 0:
            print("  Iteration ", fcn.n, ", Flag=", istatus, " NLL=", f[0], ", pars=",
                  sess.run(float_tfpars))

            tmp_results = {'loglh': f[0], "status": -1}
            for n, p in enumerate(float_tfpars):
                tmp_results[p.par_name] = (p.prev_value, 0.)
            # write_fit_results(tmp_results, tmp_file)

    fcn.n = 0
    minuit = TVirtualFitter.Fitter(0, len(float_tfpars))  # Create MINUIT instance
    minuit.Clear()
    minuit.SetFCN(fcn)
    arglist = array.array('d', 10 * [0])  # Auxiliary array for MINUIT parameters

    for n, p in enumerate(float_tfpars):  # Declare fit parameters in MINUIT
        #    print "passing parameter %s to Minuit" % p.par_name
        step_size = p.step_size
        lower_limit = p.lower_limit
        upper_limit = p.upper_limit
        if not step_size:
            step_size = 1e-6
        if not lower_limit:
            lower_limit = 0.
        if not upper_limit:
            upper_limit = 0.
        minuit.SetParameter(n, p.par_name, p.init_value, step_size, lower_limit, upper_limit)

    arglist[0] = 0.5
    minuit.ExecuteCommand("SET ERR", arglist, 1)  # Set error definition for neg. likelihood fit
    if use_gradient is True or use_gradient == "CHECK":
        minuit.ExecuteCommand("SET GRA", arglist, 0)  # Ask analytic gradient_par
    elif use_gradient == "FORCE":
        arglist[0] = 1
        minuit.ExecuteCommand("SET GRA", arglist, 1)  # Ask analytic gradient_par
    arglist[0] = call_limit  # Set call limit
    minuit.ExecuteCommand("MIGRAD", arglist, 1)  # Perform minimisation

    minuit.ExecuteCommand("SET NOG", arglist, 0)  # Ask no analytic gradient_par

    if run_hesse:
        minuit.ExecuteCommand("HESSE", arglist, 1)

    if run_minos:
        minuit.ExecuteCommand("MINOS", arglist, 1)

    results = {}  # Get fit results and update parameters
    for n, p in enumerate(float_tfpars):
        p.update(sess, minuit.GetParameter(n))
        p.fitted_value = minuit.GetParameter(n)
        p.error = minuit.GetParError(n)
        if run_minos:
            eplus = array.array("d", [0.])
            eminus = array.array("d", [0.])
            eparab = array.array("d", [0.])
            globcc = array.array("d", [0.])
            minuit.GetErrors(n, eplus, eminus, eparab, globcc)
            p.positive_error = eplus[0]
            p.negative_error = eminus[0]
            results[p.par_name] = (p.fitted_value, p.error, p.positive_error, p.negative_error)
        else:
            results[p.par_name] = (p.fitted_value, p.error)

    # Get status of minimisation and NLL at the minimum
    maxlh = array.array("d", [0.])
    edm = array.array("d", [0.])
    errdef = array.array("d", [0.])
    nvpar = array.array("i", [0])
    nparx = array.array("i", [0])
    fitstatus = minuit.GetStats(maxlh, edm, errdef, nvpar, nparx)

    # return fit results
    results["edm"] = edm[0]
    results["loglh"] = maxlh[0]
    results["status"] = fitstatus
    results["iterations"] = fcn.n
    return results
