import numpy as np
import tensorflow as tf

from ..gradient_clipping import GradientClipping


def test_clip_value(sess):
    lr, value = 0.2, 0.1
    x_val = 1.
    optimizer = GradientClipping(
        tf.train.GradientDescentOptimizer(lr),
        value,
        clip_by='value',
    )
    x = tf.Variable(x_val)
    y = 0.5 * x  # dy/dx = 0.5

    train_op = optimizer.minimize(y)
    sess.run(tf.variables_initializer([x]))
    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(x),
        x_val - lr * np.minimum(value, 0.5),
    )


def test_clip_norm(sess):
    lr, value = 0.2, 0.5
    x_val = np.array([3., 4.])
    optimizer = GradientClipping(
        tf.train.GradientDescentOptimizer(lr),
        value,
        clip_by='norm',
    )
    x = tf.Variable(x_val)
    y = tf.nn.l2_loss(x)  # dy/dx = x

    train_op = optimizer.minimize(y)
    sess.run(tf.variables_initializer([x]))
    sess.run(train_op)
    np.testing.assert_array_almost_equal(
        sess.run(x),
        x_val - lr * x_val * np.minimum(value / np.linalg.norm(x_val), 1.),
    )
