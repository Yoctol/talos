import tensorflow as tf

from ..beam_search_decode import beam_search_decode


def test_beam_search_decode():
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5)
    first_input = tf.zeros(shape=[7, 3], dtype=tf.float32)
    next_input_producer = tf.keras.layers.Dense(units=3)
    outputs = beam_search_decode(
        cell=cell,
        first_input=first_input,
        maxlen=10,
        beam_width=3,
        output_layer=lambda x: x,
        next_input_producer=next_input_producer,
    )
    assert outputs.shape.as_list() == [7, 10, 5]
