import tensorflow as tf

from ..radam import RAdamOptimizer


def test_radam(sess):
    radam_opt = RAdamOptimizer(0.1)
    adam_opt = tf.train.AdamOptimizer(0.1)
    with tf.variable_scope('test_radam'):
        x = tf.Variable(1.)
        update_x = radam_opt.minimize(x)  # constant grad 1
        y = tf.Variable(1.)
        update_y = adam_opt.minimize(y)  # constant grad 1

    sess.run(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test_radam')),
    )
    for _ in range(5):
        sess.run([update_x, update_y])

    x_val, y_val = sess.run([x, y])
    assert x_val > y_val  # since updates of x is warming up
