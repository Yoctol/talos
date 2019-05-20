import pytest

import numpy as np
import tensorflow as tf

from ..mask_conv import MaskConv1D


@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_mask_conv(padding, sess):
    layer = MaskConv1D(filters=1, kernel_size=3, kernel_initializer='ones', padding=padding)
    x = tf.ones([2, 5, 3], dtype=tf.float32)
    x._keras_mask = tf.sequence_mask([2, 4], maxlen=5)
    out = layer(x)

    sess.run(tf.variables_initializer(layer.variables))
    if padding == 'same':
        expected_out = [
            [[6.], [6.], [3.], [0.], [0.]],
            [[6.], [9.], [9.], [6.], [3.]],
        ]
        expected_mask = [
            [True, True, False, False, False],  # act like maxlen = 2
            [True, True, True, True, False],  # act like maxlen = 4
        ]
    else:
        expected_out = [
            [[6.], [3.], [0.]],
            [[9.], [9.], [6.]],
        ]
        expected_mask = [
            [False, False, False],  # act like maxlen = 2
            [True, True, False],  # act like maxlen = 4
        ]

    np.testing.assert_array_almost_equal(sess.run(out), expected_out, decimal=4)
    np.testing.assert_array_equal(sess.run(out._keras_mask), expected_mask)
