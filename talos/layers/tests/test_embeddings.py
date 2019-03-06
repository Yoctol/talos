import pytest

import numpy as np
import tensorflow as tf

from ..embeddings import Embedding
from talos.networks import Sequential


@pytest.fixture(scope='module')
def inputs():
    return tf.placeholder(tf.int32, shape=[None, 3])


@pytest.fixture(scope='module')
def mask(inputs):
    return tf.placeholder(tf.bool, shape=inputs.shape)


def test_output_shape(inputs):
    embed_layer = Embedding(vocab_size=3, embeddings_dim=5, mask_index=1)
    outputs = embed_layer(inputs)
    maxlen = inputs.shape[1].value
    assert outputs.shape.as_list() == [None, maxlen, 5]


@pytest.mark.parametrize('mask_index', [
    1,
    [1, 2, 3],
])
def test_mask_index(inputs, mask, sess, mask_index):
    maxlen = inputs.shape[1].value
    embed_layer = Embedding(vocab_size=10, embeddings_dim=5, mask_index=mask_index)
    seq = Sequential([embed_layer])  # to handle mask propagate
    outputs = seq(inputs, mask=mask)

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    input_val = np.random.randint(0, embed_layer.vocab_size, size=[5, maxlen])
    mask_val = np.random.randint(0, 2, size=[5, maxlen], dtype=np.bool)
    output_mask_val = sess.run(
        outputs._keras_mask,
        feed_dict={inputs: input_val, mask: mask_val},
    )
    if isinstance(mask_index, int):
        expected_val = input_val != mask_index
    else:
        expected_val = np.isin(input_val, mask_index, invert=True)

    expected_val = np.logical_and(mask_val, expected_val)

    assert np.array_equal(output_mask_val, expected_val)


@pytest.mark.parametrize('invalid_mask_index', [
    'mask',  # not int
    3.,  # not int
    ['mask'],  # not int
    6,  # > vocab_size
    [1, 2, [3]],  # nested
    -1,
])
def test_init_from_invalid_mask_index_raise(invalid_mask_index):
    with pytest.raises(ValueError):
        Embedding(vocab_size=5, embeddings_dim=5, mask_index=invalid_mask_index)


def test_construct_from_weights(inputs, sess):
    weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    embed_layer = Embedding.from_weights(weights)
    embed_layer(inputs)  # to build variables

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    weights_val = sess.run(embed_layer.embeddings)

    np.testing.assert_array_almost_equal(weights, weights_val)


@pytest.mark.parametrize('invalid_weights', [
    np.zeros([5]),
    np.zeros([1, 2, 3]),
])
def test_construct_from_invalid_weights_raise(invalid_weights):
    with pytest.raises(ValueError):
        Embedding.from_weights(invalid_weights)
