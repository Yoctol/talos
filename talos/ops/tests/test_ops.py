import pytest

import tensorflow as tf

from ..ops import sequence_reduce_mean


@pytest.yield_fixture(scope="function")
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_sequence_reduce_mean(graph):
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

    with tf.Session(graph=graph) as sess:
        assert sess.run(mean_x) == expected_mean_x
