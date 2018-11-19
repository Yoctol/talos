import pytest

import tensorflow as tf

from ..dynamic_rnn import DynamicRecurrent


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_dynamic_rnn(graph):
    cell = tf.keras.layers.GRUCell(5)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])
    layer = DynamicRecurrent(cell)
    outputs = layer(inputs)
    assert outputs.shape.as_list() == [None, 5]
