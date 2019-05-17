import pytest

import numpy as np
import tensorflow as tf

from ..mask_conv import MaskConv1D


@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_mask_conv(padding, sess):
    layer = MaskConv1D(filters=3, kernel_size=3, padding=padding)
    x = tf.ones([2, 5, 3], dtype=tf.float32)
    x._keras_mask = tf.constant([
        [True, False, True, True, True],
        [False, False, False, True, False],
    ], dtype=tf.float32)
    out = layer(x)

    sess.run(tf.variables_initializer(layer.variables))
    if padding == 'same':
        expected_mask = [
            [True, True, True, True, True],
            [False, False, True, True, True],
        ]
    else:
        expected_mask = [
            [False, False, True],
            [False, False, False],
        ]

    np.testing.assert_array_equal(
        sess.run(out._keras_mask),
        expected_mask,
    )
