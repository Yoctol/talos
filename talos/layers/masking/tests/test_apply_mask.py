import numpy as np
import tensorflow as tf

from ..apply_mask import ApplyMask


def test_apply_mask(sess):
    layer = ApplyMask()
    x = tf.constant([[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]], dtype=tf.float32)
    x._keras_mask = tf.constant([[True, False, True, True, False]])
    out = layer(x)

    assert out._keras_mask is x._keras_mask
    np.testing.assert_array_equal(
        sess.run(out),
        [[[0., 1.], [0., 0.], [4., 5.], [6., 7.], [0., 0.]]],
    )
