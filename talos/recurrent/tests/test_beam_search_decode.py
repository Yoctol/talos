import numpy as np
import tensorflow as tf

from ..beam_search_decode import beam_search_decode


def test_beam_search_decode():
    batch_size = 7
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5)
    first_input = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
    next_input_producer = tf.keras.layers.Dense(units=3)
    outputs = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=10,
        beam_width=3,
        output_layer=lambda x: x,
        next_input_producer=next_input_producer,
    )
    assert outputs.shape.as_list() == [batch_size, 10, 5]


def test_beam_search_decode_dynamic():
    graph = tf.Graph()
    with graph.as_default():
        batch_size = 2
        cell = tf.nn.rnn_cell.GRUCell(num_units=5)
        first_input = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
        init_state = tf.placeholder(shape=[batch_size, 5], dtype=tf.float32)
        next_input_producer = tf.keras.layers.Dense(units=3)
        output_tensor = beam_search_decode(
            cell=cell,
            first_input=first_input,
            maxlen=10,
            beam_width=3,
            output_layer=lambda x: x,
            next_input_producer=next_input_producer,
            init_state=init_state,
        )

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        )
        outputs = sess.run(
            output_tensor,
            feed_dict={
                first_input: np.zeros([2, 3], dtype=np.float32),
                init_state: np.zeros([2, 5], dtype=np.float32),
            }
        )
    assert outputs.shape == (2, 10, 5)
