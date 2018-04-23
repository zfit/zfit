from __future__ import print_function, division, absolute_import

import array  # TODO: needed? numpy array?

import tensorflow as tf
import numpy as np
from ROOT import TVirtualFitter, TNtuple, TH1, TH2, TH3

from zfit.core import tfext
from zfit.core.math import interpolate
from zfit.settings import fptype

cacheable_tensors = []


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
                tf.constant([hist.GetXaxis().GetBinCenter(1)], dtype=fptype),
                tf.constant([hist.GetXaxis().GetBinCenter(nx)], dtype=fptype),
                ]
            for x in range(nx):
                data[x] = hist.GetBinContent(x + 1)
            self.ns = tf.constant([nx - 1], dtype=fptype)

        if isinstance(hist, TH2):
            nx = hist.GetNbinsX()
            ny = hist.GetNbinsY()
            data = np.zeros((nx, ny), dtype=np.dtype('d'))
            self.limits = [
                tf.constant([hist.GetXaxis().GetBinCenter(1), hist.GetYaxis().GetBinCenter(1)],
                            dtype=fptype),
                tf.constant([hist.GetXaxis().GetBinCenter(nx), hist.GetYaxis().GetBinCenter(ny)],
                            dtype=fptype),
                ]
            for x in range(nx):
                for y in range(ny):
                    data[x][y] = hist.GetBinContent(x + 1, y + 1)
            self.ns = tf.constant([nx - 1, ny - 1], dtype=fptype)

        if isinstance(hist, TH3):
            nx = hist.GetNbinsX()
            ny = hist.GetNbinsY()
            nz = hist.GetNbinsZ()
            data = np.zeros((nx, ny, nz), dtype=np.dtype('d'))
            self.limits = [
                tf.constant([hist.GetXaxis().GetBinCenter(1), hist.GetYaxis().GetBinCenter(1),
                             hist.GetZaxis().GetBinCenter(1)], dtype=fptype),
                tf.constant([hist.GetXaxis().GetBinCenter(nx), hist.GetYaxis().GetBinCenter(ny),
                             hist.GetZaxis().GetBinCenter(nz)], dtype=fptype),
                ]
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        data[x][y][z] = hist.GetBinContent(x + 1, y + 1, z + 1)
            self.ns = tf.constant([nx - 1, ny - 1, nz - 1], dtype=fptype)

        self.array = tf.constant(data, dtype=fptype)

    def shape(self, x):
        """
          Method that returns a TF graph with the interpolation result for for a set of N
          M-dimensional points
            x - TF tensor of shape (N, M)
        """
        c = (x - self.limits[0]) / (self.limits[1] - self.limits[0]) * self.ns
        return interpolate(self.array, c)


def accept_reject_sample(density, sample):
    """
      Return toy MC sample graph using accept-reject method
        density : function to calculate density
        sample  : input uniformly distributed sample
    """
    x = sample[:, 0:-1]
    if density:
        r = sample[:, -1]
        return tf.boolean_mask(x, density(x) > r)
    else:
        return x


# -- modified to (optionally) introduce vetoed-window -- #
def create_accept_reject_sample(sess, density, x, sample, veto_min, veto_max):
    """
      Create toy MC sample using accept-reject method for a density defined as a graph
        sess    : Tf session
        density : density graph
        x       : phase space placeholder used for density graph definition
        sample  : input uniformly distributed sample
        veto_min: low q2 of the resonance veto, ignored if = 0
        veto_max: high q2 of the resonance veto, ignored if = 0
      Returns numpy array of generated points
    """
    p = sample[:, 0:-1]
    r = sample[:, -1]
    q2 = sample[:, 3]
    pdf_data = sess.run(density, feed_dict={x: p})
    if veto_max == 0 and veto_min == 0:
        return p[pdf_data > r]
    else:
        return p[np.all([pdf_data > r, np.any([q2 < veto_min, q2 > veto_max], axis=0)], axis=0)]


def maximum_estimator(density, phsp, size):
    """
      Return the graph for the estimator of the maximum of density function
        density : density function
        phsp : phase space object (should have uniform_sample method implemented)
        size : size of the random sample for maximum estimation
    """
    sample = phsp.uniform_sample(size)
    return tf.reduce_max(density(sample))


def estimate_maximum(sess, pdf, x, norm_sample):
    """
      Estimate the maximum of density function defined as a graph
        sess : TF session
        pdf  : density graph
        x    : phase space placeholder used for the definition of the density function
        size : size of the random sample for maximum estimation
      Returns the estimated maximum of the density
    """
    pdf_data = sess.run(pdf, {x: norm_sample})
    return np.nanmax(pdf_data)


def integral(pdf, weight_func=None):
    """Return the graph for the (weighted) integral of the PDF.
        pdf : PDF
        weight_func : weight function
    """
    if weight_func:
        pdf *= weight_func
    return tf.reduce_mean(pdf)


def unbinned_NLL(pdf, integral, weight_func=None):
    """Return unbinned negative log likelihood graph for a PDF
       pdf      : PDF
       integral : precalculated integral
       weight_func : weight function

    """
    normed_log = tf.log(pdf / integral)
    if weight_func:
        normed_log *= weight_func
    return -tf.reduce_sum(normed_log)


def extended_unbinned_NLL(pdfs, integrals, n_obs, nsignals,
                          param_gauss=None, param_gauss_mean=None, param_gauss_sigma=None,
                          log_multi_gauss=None):
    """
    Return unbinned negative log likelihood graph for a PDF
    pdfs       : concatenated array of several PDFs (different regions/channels)
    integrals  : array of precalculated integrals of the corresponding pdfs
    n_obs       : array of observed num. of events, used in the extended fit and in the
    normalization of the pdf
                 (needed since when I concatenate the pdfs I loose the information on how many
                 data points are fitted with the pdf)
    nsignals   : array of fitted number of events resulted from the extended fit (function of the
    fit parameters, prop to BR)
    param_gauss : list of parameter to be gaussian constrained (CKM pars, etc.)
    param_gauss_mean : mean of parameter to be gaussian constrained
    param_gauss_sigma : sigma parameter to be gaussian constrained
    log_multi_gauss : log of the multi-gaussian to be included in the Likelihood (FF & alphas)
    """
    # tf.add_n(log(pdf(x))) - tf.add_n(Nev*Norm)
    nll = - (tf.reduce_sum(tf.log(pdfs)) - tf.reduce_sum(
        tf.cast(n_obs, tf.float64) * tf.log(integrals)))

    # Extended fit to number of events
    nll += - tf.reduce_sum(-nsignals + tf.cast(n_obs, tf.float64) * tf.log(nsignals))

    # gaussian constraints on parameters (CKM) # tf.add_n( (par-mean)^2/(2*sigma^2) )
    if param_gauss is not None:
        nll += tf.reduce_sum(
            tf.square(param_gauss - param_gauss_mean) / (2. * tf.square(param_gauss_sigma)))

    # multivariate gaussian constraints on param that have correlations (alphas, FF)
    if log_multi_gauss is not None:
        nll += - log_multi_gauss

    return nll


# -- modified to contain all gauss constr., extended fit to Nevents not included -- #
def unbinned_angular_NLL(pdfs, integrals, nevents,
                         param_gauss=None, param_gauss_mean=None, param_gauss_sigma=None,
                         log_multi_gauss=None):
    """
    Return unbinned negative log likelihood graph for a PDF
    pdfs       : concatenated array of several PDFs (different regions/channels)
    integrals  : array of precalculated integrals of the corresponding pdfs
    nevents    : array of num. of events in the dataset to be fitted to the corresponding pdfs
                 (needed since when I concatenate the pdfs I loose the information on how many
                 data points are fitted with the pdf)
    param_gauss : list of parameter to be gaussian constrained (CKM pars, etc.)
    param_gauss_mean : mean of parameter to be gaussian constrained
    param_gauss_sigma : sigma parameter to be gaussian constrained
    log_multi_gauss : log of the multi-gaussian to be included in the Likelihood (FF & alphas)
    """
    # tf.add_n(log(pdf(x))) - tf.add_n(Nev*Norm)
    nll = - (tf.reduce_sum(tf.log(pdfs)) - tf.reduce_sum(
        tf.cast(nevents, tf.float64) * tf.log(integrals)))

    # gaussian constraints on parameters (CKM) # tf.add_n( (par-mean)^2/(2*sigma^2) )
    if param_gauss is not None:
        nll += tf.reduce_sum(
            tf.square(param_gauss - param_gauss_mean) / (2. * tf.square(param_gauss_sigma)))

    # multivariate gaussian constraints on param that have correlations (alphas, FF)
    if log_multi_gauss is not None:
        nll += - log_multi_gauss

    return nll


# -- modified to contain extended fit to Nevents + all gauss constr.,  pdf not included (BR only)
#  -- #
def extended_NLL_poiss(nsignals, n_obs, param_gauss, param_gauss_mean, param_gauss_sigma,
                       log_multi_gauss):
    # Extended fit to number of events
    nll = - tf.reduce_sum(-nsignals + tf.cast(n_obs, tf.float64) * tf.log(nsignals))

    # gaussian constraints on parameters (CKM) # tf.add_n( (par-mean)^2/(2*sigma^2) )
    nll += tf.reduce_sum(
        tf.square(param_gauss - param_gauss_mean) / (2. * tf.square(param_gauss_sigma)))

    # multivariate gaussian constraints on param that have correlations (alphas, FF)
    nll += - log_multi_gauss

    return nll


def switches(size):
    """
      Create the list of switches (flags that control the components of the PDF for use with e.g.
      fit fractions)
        size : number of components of the PDF
    """
    p = [tf.placeholder_with_default(tfext.constant(1.), shape=()) for _ in range(size)]
    return p


# -- modified to (optionally) introduce vetoed-window -- #
def run_toy_MC(sess, pdf, x, phsp, size, majorant, chunk=200000, switches=None, seed=None,
               veto_min=0.0, veto_max=0.0):
    """
      Create toy MC sample. To save memory, the sample is generated in "chunks" of a fixed size
      inside
      TF session, which are then concatenated.
        sess : TF session
        pdf : PDF graph
        x   : phase space placeholder used for PDF definition
        phsp : phase space
        size : size of the target data sample (if >0) or number of chunks (if <0)
        majorant : maximum PDF value for accept-reject method
        chunk : chunk size
        switches : optional list of switches for component weights
        veto_min : low q2 of the resonance veto, default is 0
        veto_max : high q2 of the resonance veto, default is 0
    """
    first = True
    length = 0
    nchunk = 0

    phsp_sample = phsp.filter(x)

    if seed:
        np.random.seed(seed)
    while length < size or nchunk < -size:
        initsample = phsp.unfiltered_sample(chunk, majorant)
        d1 = sess.run(phsp_sample, feed_dict={x: initsample})
        d = create_accept_reject_sample(sess, pdf, x, d1, veto_min, veto_max)
        if switches:
            weights = []
            v = sess.run(pdf, feed_dict={x: d})
            for i in range(len(switches)):
                fdict = {}
                for j in range(len(switches)):
                    fdict[switches[j]] = 0.
                fdict[switches[i]] = 1.
                fdict[x] = d
                v1 = sess.run(pdf, feed_dict=fdict)
                weights += [v1 / v]
            d = np.append(d, np.transpose(np.array(weights, dtype=np.dtype('f'))), axis=1)

        if first:
            data = d
        else:
            data = np.append(data, d, axis=0)
        first = False
        length += len(d)
        nchunk += 1
        print("  Chunk {:d}, size={:d}, total length={:d}".format(nchunk, len(d), length))

    if size > 0:
        return data[:size]
    else:
        return data


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
      Return a numpy tuple_array with the values from TNtuple.
        ntuple : input TNtuple
        variables : list of ntuple variables to read
    """
    data = []  # TODO: smell, unused?
    code_list = []
    for v in variables:
        code_list += [compile("i.{}".format(v), '<string>', 'eval')]
    nentries = ntuple.GetEntries()
    nvars = len(variables)
    tuple_array = np.zeros((nentries, nvars))
    for n, i in enumerate(ntuple):
        for m, v in enumerate(code_list):
            tuple_array[n][m] = eval(v)
        if n % 100000 == 0:
            print(n, "/", nentries)
    return tuple_array


def gradient(func):
    """
      Returns TF graph for analytic gradient of the input function wrt all floating variables
    """
    tfpars = tf.trainable_variables()  # Create TF variables
    float_tfpars = [p for p in tfpars if p.floating()]  # List of floating parameters
    return tf.gradients(func, float_tfpars)  # Get analytic gradient


def load_data(sess, phsp, name, data):
    """
      Load data to TF machinery and return a TF variable corresponding to it
      (to be further used for model fitting).
        sess   : TF session
        phsp   : phase space object for data
        name   : name for the variable and placeholder
        data   : 2D numpy array with data to be loaded
        return value : TF variable containing data
    """
    placeholder = phsp.placeholder(name)
    shape = data.shape
    variable = tf.get_variable(name, shape=shape, dtype=fptype,
                               initializer=tf.constant_initializer(0.0), trainable=False)
    initializer = variable.assign(placeholder)
    sess.run(initializer, feed_dict={placeholder: data})
    return variable


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
        gradient     : external gradient graph. If None and use_gradient is not False, will be
                       calculated internally
        use_gradient  : flag to control the use of analytic gradient while fitting:
                       None or False   : gradient is not used
                       True or "CHECK" : analytic gradient will be checked with finite elements,
                                        and will be used is they match
                       "FORCE"         : analytic gradient will be used regardless.
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
        gradient = tf.gradients(nll, float_tfpars)  # Get analytic gradient

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

        if istatus == 2:  # If gradient calculation is needed
            dnll = sess.run(gradient, feed_dict=feeds, options=options,
                            run_metadata=run_metadata)  # Calculate analytic gradient
            for j in range(len(float_tfpars)):
                gin[j] = dnll[j]  # Pass gradient to MINUIT
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
    if use_gradient == True or use_gradient == "CHECK":
        minuit.ExecuteCommand("SET GRA", arglist, 0)  # Ask analytic gradient
    elif use_gradient == "FORCE":
        arglist[0] = 1
        minuit.ExecuteCommand("SET GRA", arglist, 1)  # Ask analytic gradient
    arglist[0] = call_limit  # Set call limit
    minuit.ExecuteCommand("MIGRAD", arglist, 1)  # Perform minimisation

    minuit.ExecuteCommand("SET NOG", arglist, 0)  # Ask no analytic gradient

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


def initial_values():
    """
      Return initial values of free parameters in the same structure
      as for the fit result.
    """
    tfpars = tf.trainable_variables()  # Create TF variables
    float_tfpars = [p for p in tfpars if p.floating()]  # List of floating parameters
    results = {}
    for n, p in enumerate(float_tfpars):
        results[p.par_name] = (p.init_value, p.step_size)
    results["loglh"] = 0.
    results["status"] = 0
    return results


def write_fit_results(results, BR_names, BR_fit, BR_gen, filename):
    """
      Write the dictionary of fit results to text file
        results : fit results as returned by MinuitFit
        filename : file name
    """
    tfpars = tf.trainable_variables()  # Create TF variables
    float_tfpars = [p for p in tfpars if p.floating()]
    with open(filename, "w") as f:
        for p in float_tfpars:
            s = "{} ".format(p.par_name)
            for i in results[p.par_name]:
                s += "{:f} ".format(i)
            print(s)

            f.write(s + "\n")
        for i in range(len(BR_fit)):
            s = BR_names[i] + " {:.15f} {:.15f}".format(BR_fit[i], BR_gen[i])
            print(s)
            f.write(s + "\n")
        s = "loglh {:f} {:d}".format(results["loglh"], results["status"])
        print(s)
        f.write(s + "\n")


def read_fit_results(sess, filename):
    """
      Read the dictionary of fit results from text file
        sess     : TF session
        filename : file name
    """
    print("Reading results from ", filename)
    tfpars = tf.trainable_variables()  # Create TF variables
    float_tfpars = [p for p in tfpars if p.floating()]
    par_dict = {}
    float_par_dict = {}
    for i in float_tfpars:
        float_par_dict[i.par_name] = i
    for i in tfpars:
        par_dict[i.par_name] = i
    with open(filename, "r") as f:
        for l in f:
            ls = l.split()
            name = ls[0]
            value = float(ls[1])
            error = float(ls[2])
            if name in par_dict.keys():
                # print name, " = ", value
                par_dict[name].update(sess, value)
                par_dict[name].init_value = value
                if name in float_par_dict.keys():
                    par_dict[name].step_size = error / 10.


def calc_fit_fractions(sess, pdf, x, switches, norm_sample):
    """
      Calculate fit fractions for PDF components
        sess        : TF session
        pdf         : PDF graph
        x           : phase space placeholder used for PDF definition
        switches    : list of switches
        norm_sample : normalisation sample. Not needed if external integral is provided
    """
    pdf_norm = sess.run(pdf, feed_dict={x: norm_sample})
    total_int = np.sum(pdf_norm)
    fit_fractions = []
    for i in range(len(switches)):
        fdict = {}
        for j in range(len(switches)):
            fdict[switches[j]] = 0.
        fdict[switches[i]] = 1.
        fdict[x] = norm_sample
        pdf_norm = sess.run(pdf, feed_dict=fdict)
        part_int = np.sum(pdf_norm)
        fit_fractions += [part_int / total_int]
    return fit_fractions


def calc_CP_fit_fraction(sess, pdf_particle, pdf_antiparticle, x, switches, norm_sample):
    """
      Calculate CPC and CPV fit fractions for PDF components
        sess              : TF session
        pdf_particle      : PDF of particle decay
        pdf_antiparticle  : PDF of anti-particle decay
        x                 : phase space placeholder used for PDF definition
        switches          : list of switches
        norm_sample       : normalisation sample. Not needed if external integral is provided
    """

    norm_part = np.sum(sess.run(pdf_particle, feed_dict={x: norm_sample}))
    norm_anti = np.sum(sess.run(pdf_antiparticle, feed_dict={x: norm_sample}))

    integral = norm_part + norm_anti
    cpv_int = norm_part - norm_anti  # TODO: smell, unused?

    cpc_fit_fractions = []
    cpv_fit_fractions = []
    for i in range(len(switches)):
        fdict = {x: norm_sample}
        for j in range(len(switches)):
            fdict[switches[j]] = 0.
        fdict[switches[i]] = 1.

        norm_part = np.sum(sess.run(pdf_particle, feed_dict=fdict))
        norm_anti = np.sum(sess.run(pdf_antiparticle, feed_dict=fdict))

        cpc_fit_fractions += [(norm_part + norm_anti) / integral]
        cpv_fit_fractions += [(norm_part - norm_anti) / integral]
    return cpc_fit_fractions, cpv_fit_fractions


def write_fit_fractions(fit_fractions, names, filename):
    """
      Write fit fractions to text file
        fit_fractions : list of fit fractions returned by FitFractions
        names : list of component names
        filename : file name
    """
    with open(filename, "w") as f:
        sum_fit_fractions = 0
        for n, ff in zip(names, fit_fractions):
            s = "{} {:f}".format(n, ff)
            print(s)
            f.write(s + "\n")
            sum_fit_fractions += ff
        s = "Sum {:f}".format(sum_fit_fractions)
        print(s)
        f.write(s + "\n")
