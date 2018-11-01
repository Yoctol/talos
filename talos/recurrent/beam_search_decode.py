from typing import Callable

import tensorflow as tf

from .dynamic_decode import RECURRENT_CELL_TYPE, _get_init_state


def beam_search_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        beam_width: int,
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
                "Cell should support `zero_state` or `state_size`."
                "otherwise, `init_state` should be given")

    batch_size = first_input.shape.as_list()[0]
    top_k_logprob = tf.log(
        tf.one_hot(
            tf.zeros([batch_size], dtype=tf.int32),
            depth=beam_width,
        )
    )  # shape (N, k)
    inputs = _tile_at_first_axis(first_input, beam_width)  # shape (Nk, d_i)
    state = _nested_tile_at_first_axis(init_state, beam_width)  # shape (Nk, d_s)
    top_k_output = None
    offset = tf.expand_dims(
        tf.range(0, batch_size * beam_width, beam_width),
        axis=1,
    )  # shape (N, 1): 0, k, 2k....

    for t in range(maxlen):
        cell_output, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        logits = output_layer(cell_output)  # shape (Nk, V)
        num_classes = logits.shape.as_list()[-1]  # V
        new_logprob = tf.reshape(
            tf.reshape(
                top_k_logprob,
                shape=[-1, 1],
            ) + tf.nn.log_softmax(logits),  # shape (Nk, 1) + (Nk, V) -> (Nk, V)
            shape=[-1, beam_width * num_classes],
        )  # shape (N, kV)
        top_k_logprob, new_ids = tf.nn.top_k(
            new_logprob,
            k=beam_width,
        )  # both shape (N, k), new_ids in range [0, kV)
        # which beam from the last step to select.
        beam_ids = new_ids // num_classes  # shape (N, k), in range [0, k)
        # flatten the indices to fit tf.gather
        flatten_ids = tf.reshape(
            offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
            shape=[-1],
        )  # shape (Nk): [0, k) * k, [k, 2k) * k....
        # select beam from logits
        logits = tf.gather(logits, flatten_ids)

        if top_k_output is not None:
            top_k_output = tf.gather(top_k_output, flatten_ids)
            top_k_output = tf.concat(
                [top_k_output, tf.expand_dims(logits, axis=1)],
                axis=1,
            )
        else:
            top_k_output = tf.expand_dims(logits, axis=1)

        inputs = next_input_producer(logits)  # shape (N, d_i)
        state = _nested_gather(state, flatten_ids)

    output_ids = tf.squeeze(offset)
    output_tensor = tf.gather(top_k_output, output_ids)

    return output_tensor


def _nested_tile_at_first_axis(inputs, k):
    if hasattr(inputs, '__len__'):
        return [_tile_at_first_axis(x, k) for x in inputs]
    return _tile_at_first_axis(inputs, k)


def _tile_at_first_axis(inputs, k):
    batch_size = inputs.shape.as_list()[0]
    return tf.reshape(
        tf.tile(
            tf.expand_dims(inputs, axis=1),
            [1, k, 1],
        ),
        shape=[batch_size * k, -1],
    )


def _nested_gather(params, indices):
    if hasattr(params, '__len__'):
        return [tf.gather(p, indices) for p in params]
    return tf.gather(params, indices)
