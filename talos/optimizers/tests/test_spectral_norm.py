import pytest

import numpy as np
import tensorflow as tf

from ..spectral_norm import SpectralWeightDecay


def test_spectral_weight_decay_apply_low_rank_by_default(sess):
    lr, decay_rate = 0.2, 0.1
    x_val = 2.
    optimizer = SpectralWeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
    )
    x = tf.Variable(x_val, name='x')  # rank 0
    y = tf.pow(x, 3)  # dy/dx = 3x^2
    train_op = optimizer.minimize(y, var_list=[x])

    sess.run(tf.variables_initializer([x]))
    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(x),
        x_val - lr * 3 * (x_val ** 2),
    )


@pytest.mark.parametrize('shape', [
    [3, 4],
    [3, 4, 5],
])
def test_spectral_weight_decay(shape, sess):
    lr, decay_rate = 0.2, 0.1
    optimizer = SpectralWeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
    )

    W = tf.Variable(np.random.rand(*shape), name='kernel')
    y = tf.reduce_sum(W)  # dy/dx = 1
    train_op = optimizer.minimize(y, var_list=[W])
    u = optimizer.get_slot(W, 'u')

    assert u.shape.as_list() == [np.prod(shape[:-1])]

    sess.run(tf.variables_initializer([W, u]))
    W_val, u_val = sess.run([W, u])
    v_val = W_val.reshape([-1, shape[-1]]).T @ u_val
    v_val /= np.linalg.norm(v_val)
    decay_val = decay_rate * np.expand_dims(W_val @ v_val, -1) * v_val

    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(W),
        W_val - decay_val - lr * 1.,
    )
