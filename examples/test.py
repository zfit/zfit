# from __future__ import print_function, division, absolute_import
#
# # deactivating CUDA capable gpus
# suppress_gpu = False
# if suppress_gpu:
#     import os
#
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     print("CUDA capable GPUs purposely deactivated.")
#
# import tensorflow as tf
# import tensorflow_probability.python.mcmc as mc
#
# import sys
#
# sys.path.append("../")
#
# from zfit.core import optimization as opt
# from zfit.core import kinematics as kin
# from zfit.physics import constants as const
# from zfit.physics.flavour.rare_decays.bToKstarll import decay_rates as dec
#
# if __name__ == "__main__":
#
#     phsp = kin.FourBodyAngularPhaseSpace(1., 8., )
#
#
#     ### Start of model description
#
#     def model(x):
#         return dec.d4_gamma(phsp, x, const.mMu)
#
#
#     ### End of model description
#
#     data_ph = phsp.data_placeholder
#     norm_ph = phsp.norm_placeholder
#
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#
#         sample_size = 1000000
#         # print(phsp.uniform_sample(sample_size))
#         # print(mc.sample_halton_sequence(6, num_results=sample_size, seed=42))
#         uniform_sample = (mc.sample_halton_sequence(6, num_results=sample_size) - 0.5) * 100.
#         # uniform_sample = phsp.uniform_sample(sample_size)
#         norm_sample = sess.run(uniform_sample)
#         majorant = opt.estimate_maximum(sess, model(data_ph), data_ph, norm_sample) * 1.1
#         print("Maximum = ", majorant)
#
#         data_sample = opt.run_toy_MC(sess, model(data_ph), data_ph, phsp, 10000, majorant,
#                                      chunk=sample_size)
#
#         norm = opt.integral(model(norm_ph))
#         nll = opt.unbinned_NLL(model(data_ph), norm)
#
#         result = opt.run_minuit(sess, nll, {data_ph: data_sample, norm_ph: norm_sample},
#                                 run_minos=True)
#     print(result)
#     # opt.write_fit_results(result, "result.txt")
