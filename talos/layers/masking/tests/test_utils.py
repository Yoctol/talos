import numpy as np
import tensorflow as tf

from ..utils import apply_mask


def test_apply_mask(sess):
    x = tf.constant([[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]], dtype=tf.float32)
    assert apply_mask(x) is x

    mask = tf.constant([[True, False, True, True, False]])
    out = apply_mask(x, mask)

    np.testing.assert_array_equal(
        sess.run(out),
        [[[0., 1.], [0., 0.], [4., 5.], [6., 7.], [0., 0.]]],
    )
