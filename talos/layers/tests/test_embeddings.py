import pytest

import numpy as np
import tensorflow as tf

from ..embeddings import Embedding


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_mask_shape(graph):
    inputs = tf.placeholder(tf.int32, shape=[None, 3])
    embed_layer = Embedding(vocab_size=3, embeddings_dim=5, mask_index=1)
    outputs = embed_layer(inputs)
    assert outputs.shape.as_list() == [None, 3, 5]

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        mask_val = sess.run(
            outputs._keras_mask,
            feed_dict={inputs: [[0, 1, 2]]},
        )

    assert np.array_equal(mask_val, np.array([[True, False, True]]))


def test_construct_from_weights(graph):
    weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    inputs = tf.placeholder(tf.int32, shape=[None, 3])
    embed_layer = Embedding.from_weights(weights)
    embed_layer(inputs)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        weights_val = sess.run(embed_layer.embeddings)

    np.testing.assert_array_almost_equal(weights, weights_val)


def test_construct_from_invalid_weights(graph):
    rank1_weights = np.random.rand(5)
    rank3_weights = np.random.rand(1, 2, 3)
    with pytest.raises(ValueError):
        Embedding.from_weights(rank1_weights)
    with pytest.raises(ValueError):
        Embedding.from_weights(rank3_weights)
