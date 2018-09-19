from __future__ import print_function, division, absolute_import

# deactivating CUDA capable gpus
suppress_gpu = False
if suppress_gpu:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("CUDA capable GPUs purposely deactivated.")

import tensorflow as tf

import sys

sys.path.append("../")

from zfit.core import optimization as opt
from zfit.core import kinematics as kin
from zfit.core.pdf import BaseDistribution
from zfit.physics import constants as const
from zfit.physics.flavour.rare_decays.bToKstarll import decay_rates as dec

if __name__ == "__main__":

    phsp = kin.FourBodyAngularPhaseSpace(1., 8., )



    ### Start of model description

    def model(x):
        return dec.d4_gamma(phsp, x, const.mMu)


    ### End of model description

    data_ph = phsp.data_placeholder


    class TestPDF(BaseDistribution):

        def _func(self, value):
            return model(value)

        def _normalization_sampler(self):
            phsp.sample = phsp.uniform_sample
            return phsp

    test_model = TestPDF()

    norm_ph = phsp.norm_placeholder

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session() as sess:
        sess.run(init)

        sample_size = 10000
        norm_sample = sess.run(phsp.uniform_sample(sample_size))
        majorant = opt.estimate_maximum(sess, model(data_ph), data_ph, norm_sample) * 1.1
        print("Maximum = ", majorant)

        data_sample = opt.run_toy_MC(sess, test_model.prob(data_ph), data_ph, phsp, 10000, majorant,
                                     chunk=sample_size)

        # norm = opt.integral(test_model.prob(norm_ph))
        norm = 1.
        nll = opt.unbinned_NLL(test_model.prob(data_ph), norm)

        result = opt.run_minuit(sess, nll, {data_ph: data_sample, norm_ph: norm_sample},
                                run_minos=True)
    print(result)
    # opt.write_fit_results(result, "result.txt")
