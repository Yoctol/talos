from typing import Callable, Tuple

import tensorflow as tf

RECURRENT_CELL_TYPE = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def _get_init_state(cell, inputs):
    batch_size = inputs.shape.as_list()[0]
    assert batch_size is not None
    dtype = inputs.dtype
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
        init_state = _get_init_state(cell, first_input)

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
