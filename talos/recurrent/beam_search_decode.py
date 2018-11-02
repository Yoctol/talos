from typing import Callable

import tensorflow as tf

from .chain_decode import RECURRENT_CELL_TYPE, get_default_init_state


_TIME_AXIS = 1


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
        init_state = get_default_init_state(cell, batch_size, first_input.dtype)

    inputs = first_input  # shape (N, d_i)
    state = init_state  # shape (N, d_s)
    beam_score = beam_logits = beam_class_ids = None
    offset = tf.expand_dims(
        tf.range(batch_size) * beam_width,
        axis=1,
    )  # shape (N, 1): 0, k, 2k....

    for t in range(maxlen):
        logits, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        n_classes = logits.shape[-1].value
        beam_score, expanded_ids = select_beam(beam_score, logits, beam_width, n_classes)
        # both shape (N, k), expanded_ids in range [0, kV)
        if t > 0:
            beam_ids = expanded_ids // n_classes  # shape (N, k), in range [0, k)
            class_ids = expanded_ids % n_classes  # shape (N, k) in range [0, V)
            # HACK flatten the indices to fit tf.gather
            flatten_ids = tf.reshape(
                offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
                shape=[-1],
            )  # shape (Nk, ): k elements in [0, k), k elements in [k, 2k)...
            logits, state, beam_logits, beam_class_ids = nested_call(
                func=tf.gather,
                nested_var=(logits, state, beam_logits, beam_class_ids),
                indices=flatten_ids,
            )
        else:
            class_ids = expanded_ids  # shape (N, k) in range [0, V)
            logits, state = nested_call(
                func=tile_batch,
                nested_var=(logits, state),
                multiplier=beam_width,
            )

        class_ids = tf.reshape(class_ids, shape=[-1])
        beam_logits = concat_on_time_axis(beam_logits, logits)
        beam_class_ids = concat_on_time_axis(beam_class_ids, class_ids)

        inputs = next_input_producer(logits, class_ids)  # shape (N, d_i)

    output_ids = tf.squeeze(offset)  # shape (N, )
    output_logits = tf.gather(beam_logits, output_ids)
    output_class_ids = tf.gather(beam_class_ids, output_ids)

    return output_logits, output_class_ids


def select_beam(beam_score, new_logits, beam_width, n_classes):
    new_score = tf.nn.log_softmax(new_logits)  # shape (Nk, V)
    if beam_score is None:
        expanded_score = new_score  # shape (N, V)
    else:
        # for broadcast: (Nk, 1) + (Nk, V) -> (Nk, V)
        expanded_score = tf.reshape(beam_score, shape=[-1, 1]) + new_score
        # expansion on last dimension (Nk, V) -> (N, kV)
        expanded_score = tf.reshape(expanded_score, shape=[-1, beam_width * n_classes])
    return tf.nn.top_k(expanded_score, k=beam_width)


def concat_on_time_axis(tensor, new_tensor):
    new_tensor = tf.expand_dims(new_tensor, axis=_TIME_AXIS)
    if tensor is None:
        return new_tensor
    return tf.concat([tensor, new_tensor], axis=_TIME_AXIS)  # shape (Nk, t, V)


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
