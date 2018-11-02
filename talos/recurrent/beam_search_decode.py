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
        init_state = _get_init_state(cell, first_input)

    inputs = first_input  # shape (N, d_i)
    state = init_state  # shape (N, d_s)
    top_k_logprob = None
    top_k_output = None

    batch_size = first_input.shape[0].value
    if batch_size is None:
        batch_size = tf.shape(first_input)[0]
    offset = tf.expand_dims(
        tf.range(0, batch_size) * beam_width,
        axis=1,
    )  # shape (N, 1): 0, k, 2k....

    for t in range(maxlen):
        cell_output, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        logits = output_layer(cell_output)  # shape (Nk, V)
        num_classes = logits.shape.as_list()[-1]  # V
        if top_k_logprob is not None:
            top_k_logprob = tf.reshape(
                tf.reshape(
                    top_k_logprob,
                    shape=[-1, 1],
                ) + tf.nn.log_softmax(logits),  # shape (Nk, 1) + (Nk, V) -> (Nk, V)
                shape=[-1, beam_width * num_classes],
            )  # shape (N, kV)
        else:
            top_k_logprob = tf.nn.log_softmax(logits)  # shape (N, V)
        top_k_logprob, new_ids = tf.nn.top_k(
            top_k_logprob,
            k=beam_width,
        )  # both shape (N, k), new_ids in range [0, kV)
        # which beam from the last step to select.
        if t > 0:
            beam_ids = new_ids // num_classes  # shape (N, k), in range [0, k)
            # flatten the indices to fit tf.gather
            flatten_ids = tf.reshape(
                offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
                shape=[-1],
            )  # shape (Nk,): [0, k) * k, [k, 2k) * k....
        else:
            beam_ids = tf.zeros_like(new_ids)  # select first beam at the first time
            flatten_ids = tf.reshape(beam_ids, shape=[-1])  # shape (Nk,)
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


def _nested_gather(params, indices):
    if hasattr(params, '__len__'):
        return [tf.gather(p, indices) for p in params]
    return tf.gather(params, indices)
