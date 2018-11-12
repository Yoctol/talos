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
    assert len(att_pool.losses) == 0


def test_attention_pooling_1d_with_reg_loss(graph):
    width, channel = 10, 4
    units, heads = 3, 5
    att_pool = GlobalAttentionPooling1D(units=units, heads=heads, reg_coeff=1.0)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = att_pool(inputs)

    assert outputs.shape.as_list() == [None, heads, channel]

    losses = att_pool.losses
    assert len(losses) == 1
    assert losses[0].shape.as_list() == []
