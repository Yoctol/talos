from typing import Callable, Tuple

import tensorflow as tf

RECURRENT_CELL_TYPE = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def _get_init_state(cell, inputs):
    batch_size = tf.shape(inputs)[0]
    dtype = cell.dtype
    if hasattr(cell, 'get_init_state'):
        init_state = cell.get_init_state(batch_size=batch_size, dtype=dtype)
    else:
        init_state = cell.zero_state(batch_size=batch_size, dtype=dtype)
    return init_state


def dynamic_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        output_layer: Callable,
        next_input_producer: Callable,
        init_state: tf.Tensor = None,
    ):
    # example of next_input_producer: lambda logit: embedding_lookup(argmax(logit))
    if init_state is None:
        try:
            init_state = _get_init_state(cell, first_input)
        except AttributeError:
            raise ValueError(
                "Cell should support get_init_state or zero_state method."
                "otherwise, init_state should be given")

    inputs = first_input
    state = init_state
    output_list = []

    for _ in range(maxlen):
        cell_output, state = cell(inputs, state)
        output = output_layer(cell_output)
        output_list.append(output)
        inputs = next_input_producer(output)  # shape (N, E)

    output_tensor = tf.stack(output_list, axis=1)  # shape (N, T-1, V)
    return output_tensor
