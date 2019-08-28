import numpy as np
import tensorflow as tf

from ..look_ahead import LookAhead


def test_look_ahead(sess):
    alpha, lr = 0.2, 0.1
    opt = LookAhead(
        tf.train.GradientDescentOptimizer(lr),
        alpha=0.2,
        explore_step=5,
    )
    with tf.variable_scope('test_look_ahead'):
        x = tf.Variable(1.)
        update_x = opt.minimize(2 * x)  # constant grad 2

    sess.run(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test_look_ahead'),
    ))
    for _ in range(4):
        sess.run(update_x)

    x_val = sess.run(x)
    np.testing.assert_almost_equal(x_val, 1. - 4 * lr * 2)
    sess.run(update_x)

    x_val = sess.run(x)
    np.testing.assert_almost_equal(
        x_val,
        1. * (1 - alpha) + (1. - 5 * lr * 2) * alpha,
    )
