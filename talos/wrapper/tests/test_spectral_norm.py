import pytest

import tensorflow as tf

from ..spectral_norm import add_spectral_norm


@pytest.fixture(scope="function")
def graph():
    return tf.Graph()


def test_spectral_norm_dense(graph):
    with graph.as_default():
        dense_layer = tf.layers.Dense(10)
        add_spectral_norm(dense_layer)

        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 20])
        dense_layer(inputs)

    assert len(dense_layer.updates) == 1
    assert dense_layer.updates[0].name == "dense/kernel/power_iter:0"
    assert dense_layer.kernel.name == "dense/kernel_sn:0"
    assert len(dense_layer.trainable_variables) == 2
    assert len(graph.get_collection(tf.GraphKeys.UPDATE_OPS)) == 1


def test_spectral_norm_gru(graph):
    with graph.as_default():
        gru_cell = tf.nn.rnn_cell.GRUCell(10)
        add_spectral_norm(gru_cell)

        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 20])
        state = tf.placeholder(dtype=tf.float32, shape=[None, gru_cell.state_size])
        gru_cell(inputs, state)

    assert len(gru_cell.updates) == 2
    assert len(gru_cell.trainable_variables) == 4
