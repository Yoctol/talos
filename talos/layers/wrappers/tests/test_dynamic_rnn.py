import pytest

import numpy as np
import tensorflow as tf

from ..dynamic_rnn import DynamicRecurrent


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_dynamic_rnn_shape(graph):
    cell = tf.keras.layers.GRUCell(5)
    layer = DynamicRecurrent(cell)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])
    seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
    # NOTE should call dynamic first
    outputs = layer(inputs, seqlen=seqlen)
    assert outputs.shape.as_list() == [None, 5]


def test_dynamic_rnn_value(graph):
    cell = tf.keras.layers.LSTMCell(5)
    static_layer = tf.keras.layers.RNN(cell, return_sequences=True)
    dynamic_layer = DynamicRecurrent(cell)

    maxlen = 5
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, maxlen, 3])
    seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
    # NOTE should call static first
    static_outputs = tf.squeeze(tf.batch_gather(
        static_layer(inputs),  # rank 3
        indices=tf.expand_dims(seqlen - 1, -1),  # to fit tf.batch_gather form
    ), axis=1)  # get the last output in seqlen window
    dynamic_outputs = dynamic_layer(inputs, seqlen=seqlen)  # rank 2

    n_samples = 10
    seqlen_batch = np.random.randint(low=2, high=maxlen + 1, size=[n_samples])
    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
        )
        static_batch, dynamic_batch = sess.run(
            [static_outputs, dynamic_outputs],
            feed_dict={
                inputs: np.random.rand(n_samples, maxlen, 3),
                seqlen: seqlen_batch,
            }
        )

    np.testing.assert_array_almost_equal(static_batch, dynamic_batch)
