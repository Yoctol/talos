import pytest

import numpy as np
import tensorflow as tf

from ..embeddings import Embedding


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_embedding(graph):
    inputs = tf.placeholder(tf.int32, shape=[None, 3])
    embed_layer = Embedding(3, 5, mask_index=1)
    outputs = embed_layer(inputs)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        mask_val = sess.run(
            outputs._keras_mask,
            feed_dict={inputs: [[0, 1, 2]]},
        )

    assert np.array_equal(mask_val, np.array([[True, False, True]]))
