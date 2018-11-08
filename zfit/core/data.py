import tensorflow as tf

if __name__ == '__main__':
    path_root = "/data/uni/b2k1ee/classification_new/2012/"
    big_root = 'Bu2KpipiEE-MC-12125000-2012-MagAll-StrippingBu2LLK.root'
    small_root = 'small.root'

    # path_root += big_root
    path_root += small_root

    import uproot


    def uproot_generator():
        for data in uproot.iterate(path=path_root, treepath='DecayTree',
                                   branches=['B_PT', "B_M"], entrysteps=4):
            yield data[b'B_PT']


    data = tf.data.Dataset.from_generator(uproot_generator,
                                          output_types=tf.float64)

    data = data.batch(batch_size=4)
    data = data.prefetch(1)

    iterator = data.make_one_shot_iterator()

    next = iterator.get_next()

    with tf.Session() as sess:
        for _ in range(5):
            print(sess.run(next))

    # directory = open_tree[]
    # directory = directory['DecayTree']
    # directory = directory['B_P']
