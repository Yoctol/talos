from unittest.mock import call

import tensorflow as tf

from ..chain_decode import chain_decode


def test_chain_decode_logic(mocker):

    def mock_cell_call(inputs, state):
        return chr(inputs + len(state)), state + 's'

    mock_cell = mocker.Mock(side_effect=mock_cell_call)
    first_input = ord('A')
    mocker.patch('tensorflow.stack', lambda lst, axis: ",".join(lst))
    output = chain_decode(
        cell=mock_cell,
        first_input=first_input,
        maxlen=5,
        next_input_producer=lambda c: ord(c),
        init_state='s',
    )
    expected_cell_calls = [
        call(first_input, 's'),
        call(first_input + 1, 'ss'),
        call(first_input + 3, 'sss'),
        call(first_input + 6, 'ssss'),
        call(first_input + 10, 'sssss'),
    ]
    expected_output = ",".join([
        chr(first_input + 1),
        chr(first_input + 3),
        chr(first_input + 6),
        chr(first_input + 10),
        chr(first_input + 15),
    ])
    mock_cell.assert_has_calls(expected_cell_calls)
    assert output == expected_output


def test_chain_decode_tf(graph):
    batch_size, maxlen = 6, 5
    cell = tf.keras.layers.LSTMCell(units=5)
    first_input = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
    init_state = (
        tf.placeholder(shape=[batch_size, cell.units], dtype=tf.float32),
        tf.placeholder(shape=[batch_size, cell.units], dtype=tf.float32),
    )
    next_input_producer = tf.keras.layers.Dense(units=3)
    outputs = chain_decode(
        cell=cell,
        first_input=first_input,
        maxlen=maxlen,
        next_input_producer=next_input_producer,
        init_state=init_state,
    )
    assert outputs.shape.as_list() == [batch_size, maxlen, cell.units]
