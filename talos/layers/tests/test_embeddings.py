import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

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


@pytest.mark.parametrize('constant', [False, True])
def test_construct_from_weights(inputs, sess, constant):
    weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    embed_layer = Embedding.from_weights(weights, constant=constant)
    embed_layer(inputs)  # to build variables

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    weights_val = sess.run(embed_layer.embeddings)

    np.testing.assert_array_almost_equal(weights_val, weights)


@pytest.mark.parametrize('constant', [False, True])
def test_auxiliary_tokens_partially_trainable(inputs, sess, constant):
    maxlen = inputs.shape[1].value
    embed_layer = Embedding.from_weights(
        np.random.uniform(size=[5, 3]).astype(np.float32),
        constant=constant,
        trainable=False,
        auxiliary_tokens=2,
    )
    word_vec = embed_layer(inputs)
    assert len(embed_layer.trainable_variables) == 1
    assert len(embed_layer.non_trainable_variables) == (0 if constant else 1)
    assert len(embed_layer.variables) == (1 if constant else 2)

    update_op = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_sum(word_vec))

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    original_weights_val = sess.run(embed_layer.total_embeddings)
    sess.run(update_op, feed_dict={inputs: np.random.choice(5 + 2, size=[10, maxlen])})
    new_weights_val = sess.run(embed_layer.total_embeddings)

    # after update:
    # first 5 row should keep
    np.testing.assert_array_almost_equal(
        original_weights_val[:5],
        new_weights_val[:5],
    )
    # others (auxiliary tokens) should change.
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(
            original_weights_val[5:],  # auxiliary tokens
            new_weights_val[5:],
        )


@pytest.mark.parametrize('constant', [False, True])
def test_extend_dims_partially_trainable(inputs, sess, constant):
    maxlen = inputs.shape[1].value
    vocab_size = 5
    original_embedding_size = 3
    embed_layer = Embedding.from_weights(
        np.random.uniform(size=[vocab_size, original_embedding_size]).astype(np.float32),
        constant=constant,
        trainable=False,
        extend_dims=2,
    )
    word_vec = embed_layer(inputs)
    assert len(embed_layer.trainable_variables) == 1
    assert len(embed_layer.non_trainable_variables) == (0 if constant else 1)
    assert len(embed_layer.variables) == (1 if constant else 2)

    update_op = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_sum(word_vec))

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    original_weights_val = sess.run(embed_layer.total_embeddings)
    sess.run(update_op, feed_dict={inputs: np.random.choice(vocab_size, size=[10, maxlen])})
    new_weights_val = sess.run(embed_layer.total_embeddings)

    # after update:
    # first 5 row should keep
    np.testing.assert_array_almost_equal(
        original_weights_val[:, : original_embedding_size],
        new_weights_val[:, : original_embedding_size],
    )
    # others (extend dims) should change.
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(
            original_weights_val[:, original_embedding_size:],  # extend dims
            new_weights_val[:, original_embedding_size:],
        )


@pytest.mark.parametrize('invalid_weights', [
    np.zeros([5]),
    np.zeros([1, 2, 3]),
])
def test_construct_from_invalid_weights_raise(invalid_weights):
    with pytest.raises(ValueError):
        Embedding.from_weights(invalid_weights)


@pytest.mark.parametrize('constant,auxiliary_tokens', [
    (True, 0),
    (True, 2),
    (False, 2),
])
def test_freeze_success(inputs, sess, constant, auxiliary_tokens):
    # build graph with constant embedding layer
    embed_layer = Embedding.from_weights(
        np.random.rand(5, 10).astype(np.float32),
        constant=constant,
        auxiliary_tokens=auxiliary_tokens,
    )
    outputs = embed_layer(inputs)
    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    # freeze graph
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=[outputs.op.name],  # node name == op name
    )

    # check frozen graph have the same embedding as before
    new_sess = create_session_from_graphdef(frozen_graph_def)
    new_inputs = new_sess.graph.get_tensor_by_name(inputs.name)
    new_outputs = new_sess.graph.get_tensor_by_name(outputs.name)

    maxlen = inputs.shape[1].value
    inputs_val = np.random.choice(embed_layer.vocab_size, size=[5, maxlen])
    outputs_val = sess.run(outputs, feed_dict={inputs: inputs_val})
    new_outputs_val = new_sess.run(new_outputs, feed_dict={new_inputs: inputs_val})

    np.testing.assert_array_almost_equal(outputs_val, new_outputs_val)


@pytest.mark.parametrize('constant,auxiliary_tokens', [
    (False, 0),  # NOTE only fail in this case
])
def test_freeze_fail(inputs, sess, constant, auxiliary_tokens):
    # build graph with variable embedding layer
    embed_layer = Embedding.from_weights(
        np.random.rand(5, 10).astype(np.float32),
        constant=constant,
        auxiliary_tokens=auxiliary_tokens,
    )
    outputs = embed_layer(inputs)

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=[outputs.op.name],  # node name == op name
    )

    with pytest.raises(ValueError):
        create_session_from_graphdef(frozen_graph_def)


def create_session_from_graphdef(graph_def):
    """
    Create new session from given tf.GraphDef object
    Arg:
       graph_def (tf.GraphDef): a tf.GraphDef object
    Return:
       session (tf.Session): a new session with given graph_def
    """
    new_sess = tf.Session(graph=tf.Graph())
    with new_sess.graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return new_sess
