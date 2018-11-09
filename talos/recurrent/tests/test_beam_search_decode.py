import pytest

import numpy as np
import tensorflow as tf

from ..beam_search_decode import beam_search_decode


@pytest.yield_fixture(scope="function")
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


@pytest.fixture(scope='function')
def cell():
    return tf.keras.layers.GRUCell(units=5)


@pytest.fixture(scope='function')
def dense_layer():
    return tf.keras.layers.Dense(units=3)


def test_beam_search_decode(graph, cell, dense_layer):
    batch_size, maxlen = 7, 5
    first_input = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
    output_logits, output_word_ids = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=maxlen,
        beam_width=3,
        next_input_producer=lambda logits, _: dense_layer(logits),
        end_token=0,
    )
    assert output_logits.shape.as_list() == [batch_size, 1, maxlen, cell.units]
    assert output_word_ids.shape.as_list() == [batch_size, 1, maxlen]


def test_beam_search_decode_dynamic(graph, cell, dense_layer):
    batch_size, maxlen = None, 5
    first_input = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
    dense_layer = tf.keras.layers.Dense(units=3)
    output_tensors = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=maxlen,
        beam_width=3,
        next_input_producer=lambda logits, _: dense_layer(logits),
        end_token=0,
    )

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
        )
        output_logits, output_word_ids = sess.run(
            output_tensors,
            feed_dict={first_input: np.zeros([2, 3], dtype=np.float32)},
        )

    assert output_logits.shape == (2, 1, maxlen, cell.units)
    assert output_word_ids.shape == (2, 1, maxlen)
