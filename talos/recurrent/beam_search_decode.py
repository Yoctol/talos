from typing import Callable

import tensorflow as tf

from .dynamic_decode import RECURRENT_CELL_TYPE, _get_init_state


def beam_search_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        beam_width: int,
        next_input_producer: Callable,
        init_state: tf.Tensor = None,
    ):
    # example of next_input_producer: lambda logit: embedding_lookup(argmax(logit))
    batch_size = first_input.shape[0].value
    if batch_size is None:
        batch_size = tf.shape(first_input)[0]
    if init_state is None:
        init_state = _get_init_state(cell, batch_size, first_input.dtype)

    inputs = first_input  # shape (N, d_i)
    state = init_state  # shape (N, d_s)
    top_k_logprob = None
    top_k_logits = None
    top_k_word_ids = None
    offset = tf.expand_dims(
        tf.range(batch_size) * beam_width,
        axis=1,
    )  # shape (N, 1): 0, k, 2k....

    for t in range(maxlen):
        logits, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        num_classes = logits.shape[-1].value  # V
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
        )  # both shape (N, k)
        # new_ids in range [0, V) (if t == 0) or [0, kV)

        if t > 0:
            beam_ids = new_ids // num_classes  # shape (N, k), in range [0, k)
            word_ids = tf.reshape(new_ids % num_classes, shape=[-1])
            # flatten the indices to fit tf.gather
            flatten_ids = tf.reshape(
                offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
                shape=[-1],
            )  # shape (Nk,): [0, k) * k, [k, 2k) * k....
            logits = tf.gather(logits, flatten_ids)
            state = nested_call(tf.gather, state, indices=flatten_ids)
            top_k_logits = time_concat_on_selected_beam(top_k_logits, logits, flatten_ids)
            top_k_word_ids = time_concat_on_selected_beam(top_k_word_ids, word_ids, flatten_ids)
        else:
            word_ids = tf.reshape(new_ids, shape=[-1])  # shape (Nk, )
            logits = tile_batch(logits, beam_width)  # shape (Nk, V)
            state = nested_call(tile_batch, state, multiplier=beam_width)  # shape (Nk, d_s)
            top_k_logits = tf.expand_dims(logits, axis=1)  # shape (Nk, 1, V)
            top_k_word_ids = tf.expand_dims(word_ids, axis=1)  # shape (Nk, 1)

        inputs = next_input_producer(logits, word_ids)  # shape (N, d_i)

    output_ids = tf.squeeze(offset)  # shape (N, )
    output_logits = tf.gather(top_k_logits, output_ids)
    output_word_ids = tf.gather(top_k_word_ids, output_ids)

    return output_logits, output_word_ids


def time_concat_on_selected_beam(tensor, new_tensor, flatten_ids):
    new_tensor = tf.expand_dims(new_tensor, axis=1)
    if tensor is None:
        return new_tensor
    tensor = tf.gather(tensor, flatten_ids)
    return tf.concat(
        [tensor, new_tensor],
        axis=1,
    )  # shape (Nk, t, V)


def nested_call(func, nested_var, *args, **kwargs):
    if hasattr(nested_var, '__len__'):
        return [nested_call(func, v, *args, **kwargs) for v in nested_var]
    return func(nested_var, *args, **kwargs)


def tile_batch(tensor, multiplier):
    shape = tensor.shape
    rank = shape.ndims
    assert rank >= 1
    tensor = tf.expand_dims(tensor, axis=1)
    tensor = tf.tile(tensor, multiples=[1, multiplier] + [1] * (rank - 1))
    return tf.reshape(tensor, shape=[-1] + shape.as_list()[1:])
