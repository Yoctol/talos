import pytest

import numpy as np
import tensorflow as tf

from ..mask_conv import MaskConv1D


@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_mask_conv(padding, sess):
    layer = MaskConv1D(filters=3, kernel_size=3, padding=padding)
    x = tf.ones([2, 5, 3], dtype=tf.float32)
    x._keras_mask = tf.sequence_mask([2, 4], maxlen=5)
    out = layer(x)

    sess.run(tf.variables_initializer(layer.variables))
    if padding == 'same':
        expected_mask = [
            [True, True, False, False, False],  # act like maxlen = 2
            [True, True, True, True, False],  # act like maxlen = 4
        ]
    else:
        expected_mask = [
            [False, False, False],  # act like maxlen = 2
            [True, True, False],  # act like maxlen = 4
        ]

    np.testing.assert_array_equal(sess.run(out._keras_mask), expected_mask)
