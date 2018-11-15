from typing import Callable, Tuple

import tensorflow as tf

RECURRENT_CELL_TYPE = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def chain_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        next_input_producer: Callable,
        init_state: tf.Tensor = None,
    ):
    # example of next_input_producer: lambda logit: embedding_lookup(argmax(logit))
    if init_state is None:
        init_state = get_default_init_state(cell, first_input)

    inputs = first_input
    state = init_state
    output_list = []

    for _ in range(maxlen):
        output, state = cell(inputs, state)
        output_list.append(output)
        inputs = next_input_producer(output)  # shape (N, E)

    output_tensor = tf.stack(output_list, axis=1)  # shape (N, T-1, V)
    return output_tensor


def get_default_init_state(cell, inputs):
    # for tf.keras.layers cell
    if hasattr(cell, 'get_initial_state'):
        return to_list(cell.get_initial_state(inputs))

    batch_size = inputs.shape[0].value
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]

    # for tf.nn.rnn_cell cell
    if hasattr(cell, 'zero_state'):
        return cell.zero_state(batch_size=batch_size, dtype=inputs.dtype)
    elif hasattr(cell, 'state_size'):
        return [
            tf.zeros([batch_size, size], dtype=inputs.dtype)
            for size in to_list(cell.state_size)
        ]
    else:
        raise ValueError(
            "Cell should support `zero_state` or `state_size`."
            "otherwise, `init_state` should be given")


def to_list(x):
    if hasattr(x, '__len__'):
        return x
    return [x]
