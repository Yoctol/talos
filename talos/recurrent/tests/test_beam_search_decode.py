import pytest

import numpy as np
import tensorflow as tf

from ..beam_search_decode import beam_search_decode


@pytest.fixture(scope='function')
def cell():
    return tf.keras.layers.GRUCell(units=5)


@pytest.fixture(scope='function')
def dense_layer():
    return tf.keras.layers.Dense(units=3)


def test_beam_search_decode(cell, dense_layer):
    batch_size, maxlen, beam_width, output_width = 7, 5, 3, 2
    first_input = tf.placeholder(shape=[batch_size, dense_layer.units], dtype=tf.float32)
    output_logits, output_word_ids = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=maxlen,
        beam_width=beam_width,
        next_input_producer=lambda logits, _: dense_layer(logits),
        end_token=0,
        output_width=output_width,
    )
    assert output_logits.shape.as_list() == [batch_size, output_width, maxlen, cell.units]
    assert output_word_ids.shape.as_list() == [batch_size, output_width, maxlen]


def test_beam_search_decode_dynamic_batch(cell, dense_layer):
    batch_size, maxlen, beam_width, output_width = None, 5, 3, 2
    first_input = tf.placeholder(shape=[batch_size, dense_layer.units], dtype=tf.float32)
    output_tensors = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=maxlen,
        beam_width=beam_width,
        next_input_producer=lambda logits, _: dense_layer(logits),
        end_token=0,
        output_width=output_width,
    )

    first_input_val = np.zeros([2] + first_input.shape.as_list()[1:], dtype=np.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_logits, output_word_ids = sess.run(
            output_tensors,
            feed_dict={first_input: first_input_val},
        )

    assert output_logits.shape == (len(first_input_val), output_width, maxlen, cell.units)
    assert output_word_ids.shape == (len(first_input_val), output_width, maxlen)
