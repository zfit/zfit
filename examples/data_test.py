# # from skhep_testdata import data_path
# import zfit
#
# path_root = "/data/uni/b2k1ee/classification_new/2012/"
# small_root = 'small.root'
# #
# path_root += small_root
# # path_root = data_path("uproot-Zmumu.root")
#
# branches = ['B_PT']  # b needed currently -> uproot
#
# data = zfit.data.Data.from_root(path=path_root, treepath='DecayTree', branches=branches)
# gauss1 = zfit.pdf.Gauss(1., 1., obs=branches)
# # lower = ((0, 0),)
# # upper = ((5, 5),)
# lower = 0
# upper = 10
# probs = gauss1.pdf(data, norm_range=zfit.Space(obs=branches, limits=(lower, upper)))
# zfit.run(probs)
# import time
# #
# # with tf.Session() as sess:
# #     # data.initialize()
# #     x = data.value()
# #
# #     for i in range(1):
# #         print(i)
# #         try:
# #             func = tf.log(x) * tf.sin(x) * tf.reduce_mean(x ** 2 - tf.cos(x) ** 2) / tf.reduce_sum(x)
# #             start = time.time()
# #             result_grad = sess.run(tf.gradients(func, x))
# #             result = sess.run(func)
# #             end = time.time()
# #             print("time needed", (end - start))
# #         except tf.errors.OutOfRangeError:
# #             print("finished at i = ", i)
# #             break
# #         print(np.shape(result))
# #         print(result[:, :10])
# #         print(result_grad)
#
# # directory = open_tree[]
# # directory = directory['DecayTree']
# # directory = directory['B_P']
