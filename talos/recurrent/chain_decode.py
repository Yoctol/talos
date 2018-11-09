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
        batch_size = first_input.shape[0].value
        if batch_size is None:
            batch_size = tf.shape(first_input)[0]
        init_state = get_default_init_state(cell, batch_size, first_input.dtype)

    inputs = first_input
    state = init_state
    output_list = []

    for _ in range(maxlen):
        output, state = cell(inputs, state)
        output_list.append(output)
        inputs = next_input_producer(output)  # shape (N, E)

    output_tensor = tf.stack(output_list, axis=1)  # shape (N, T-1, V)
    return output_tensor


def get_default_init_state(cell, batch_size, dtype):
    if hasattr(cell, 'zero_state'):
        init_state = cell.zero_state(batch_size=batch_size, dtype=dtype)
    elif hasattr(cell, 'state_size'):
        if hasattr(cell.state_size, '__len__'):
            init_state = [
                tf.zeros([batch_size, size], dtype=dtype)
                for size in cell.state_size
            ]
        else:
            init_state = [tf.zeros([batch_size, cell.state_size], dtype=dtype)]
    else:
        raise ValueError(
            "Cell should support `zero_state` or `state_size`."
            "otherwise, `init_state` should be given")
    return init_state