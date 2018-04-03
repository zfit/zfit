from __future__ import print_function, division, absolute_import

# deactivating CUDA capable gpus
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("CUDA capable GPUs purposely deactivated.")

import tensorflow as tf

import sys

sys.path.append("../")

from zfit.core import optimization as opt
from zfit.core import kinematics as kin
from zfit.physics import constants as const
from zfit.physics.flavour.rare_decays.bToKstarll import decay_rates as dec

if __name__ == "__main__":

    phsp = kin.FourBodyAngularPhaseSpace(1., 8., )


    ### Start of model description

    def model(x):
        return dec.d4Gamma(phsp, x, const.mMu)


    ### End of model description

    data_ph = phsp.data_placeholder
    norm_ph = phsp.norm_placeholder

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        sample_size = 1000000
        norm_sample = sess.run(phsp.UniformSample(sample_size))
        majorant = opt.EstimateMaximum(sess, model(data_ph), data_ph, norm_sample) * 1.1
        print("Maximum = ", majorant)

        data_sample = opt.RunToyMC(sess, model(data_ph), data_ph, phsp, 10000, majorant,
                                   chunk=sample_size)

        norm = opt.Integral(model(norm_ph))
        nll = opt.UnbinnedNLL(model(data_ph), norm)

        result = opt.RunMinuit(sess, nll, {data_ph: data_sample, norm_ph: norm_sample},
                               runMinos=True)
    print(result)
    # opt.WriteFitResults(result, "result.txt")
