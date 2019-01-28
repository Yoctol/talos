import tensorflow as tf

from ..ops import masked_reduce_mean, sequence_reduce_mean


def test_masked_reduce_mean(sess):
    x = tf.constant([
        [2, 3, 4, 5],
        [0, 2, 4, 6],
    ], dtype=tf.float32)
    mask = tf.constant([
        [True, False, True, False],
        [False, True, True, True],
    ], dtype=tf.bool)
    mean_x = masked_reduce_mean(x, mask)

    expected_mean_x = (
        (2 + 4) / 2 + (2 + 4 + 6) / 3
    ) / 2

    assert sess.run(mean_x) == expected_mean_x


def test_sequence_reduce_mean(sess):
    x = tf.constant([
        [2, 3, 4, 5],
        [0, 2, 4, 6],
        [0, 1, 2, 3],
        [5, 4, 3, 2],
    ], dtype=tf.float32)
    seqlen = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    mean_x = sequence_reduce_mean(x, seqlen)

    expected_mean_x = (
        2 + (0 + 2) / 2 + (0 + 1 + 2) / 3 + (5 + 4 + 3 + 2) / 4
    ) / 4

    assert sess.run(mean_x) == expected_mean_x
