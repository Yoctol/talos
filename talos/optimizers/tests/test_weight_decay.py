import numpy as np
import tensorflow as tf

from ..weight_decay import WeightDecay


def test_weight_decay(sess):
    lr, decay_rate = 0.2, 0.1
    x_val, z_val = 2., 1.
    optimizer = WeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
    )
    x = tf.Variable(x_val)
    z = tf.Variable(z_val)
    y = tf.pow(x, 3)  # dy/dx = 3x^2
    train_op = optimizer.minimize(y, var_list=[x])

    sess.run(tf.variables_initializer([x, z]))
    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(x),
        x_val * (1. - decay_rate) - lr * 3 * (x_val ** 2),
    )
    np.testing.assert_almost_equal(sess.run(z), z_val)  # keep since it's not updated
