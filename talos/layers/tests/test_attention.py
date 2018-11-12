import pytest

import tensorflow as tf

from ..attention import GlobalAttentionPooling1D


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_attention_pooling_1d(graph):
    width, channel = 10, 4
    units, heads = 3, 5
    att_pool = GlobalAttentionPooling1D(units=units, heads=heads)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = att_pool(inputs)

    assert outputs.shape.as_list() == [None, heads, channel]
