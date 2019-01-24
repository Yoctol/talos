import pytest

import numpy as np
import tensorflow as tf

from ..embeddings import Embedding


def test_mask_shape(sess):
    inputs = tf.placeholder(tf.int32, shape=[None, 3])
    embed_layer = Embedding(vocab_size=3, embeddings_dim=5, mask_index=1)
    outputs = embed_layer(inputs)
    assert outputs.shape.as_list() == [None, 3, 5]

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    mask_val = sess.run(
        outputs._keras_mask,
        feed_dict={inputs: [[0, 1, 2]]},
    )

    assert np.array_equal(mask_val, np.array([[True, False, True]]))


def test_construct_from_weights(sess):
    weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    inputs = tf.placeholder(tf.int32, shape=[None, 3])
    embed_layer = Embedding.from_weights(weights)
    embed_layer(inputs)

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    weights_val = sess.run(embed_layer.embeddings)

    np.testing.assert_array_almost_equal(weights, weights_val)


@pytest.mark.parametrize('invalid_weights', [
    np.zeros([5]),
    np.zeros([1, 2, 3]),
])
def test_construct_from_invalid_weights(invalid_weights):
    with pytest.raises(ValueError):
        Embedding.from_weights(invalid_weights)
