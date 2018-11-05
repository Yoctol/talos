from typing import Callable, List

import tensorflow as tf

from .chain_decode import RECURRENT_CELL_TYPE, get_default_init_state


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
    beam = Beam(batch_size=batch_size, width=beam_width)

    for _ in range(maxlen):
        output, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        output, class_ids = beam.update(observation=output)  # 更新 beam
        beam.append([output, class_ids])  # 加進 beam history 中
        state = beam.select(state)  # 幫我 tf.gather 這個東西
        inputs = next_input_producer(output, class_ids)  # shape (N, d_i)

    output_tensor, output_class_ids = beam.get_top()

    return output_tensor, output_class_ids


class Beam:

    _TIME_AXIS = 1

    def __init__(self, batch_size, width):
        self.width = width
        self.history = None
        self.score = None
        self.flatten_ids = None
        self.offset = tf.expand_dims(
            tf.range(batch_size) * width,
            axis=1,
        )  # shape (N, 1): 0, k, 2k....

    def append(self, tensors: List):
        tensors = [tf.expand_dims(t, axis=self._TIME_AXIS) for t in tensors]
        if self.history is not None:
            assert len(tensors) == len(self.history)
            self.history = [
                tf.concat([h, t], axis=self._TIME_AXIS)
                for h, t in zip(self.history, tensors)
            ]
        else:
            self.history = tensors

    def select_beam(self, observation, n_classes):
        new_score = tf.nn.log_softmax(observation)  # shape (Nk, V)
        if self.score is not None:
            # for broadcast: (Nk, 1) + (Nk, V) -> (Nk, V)
            expanded_score = tf.reshape(self.score, shape=[-1, 1]) + new_score
            # expansion on last dimension (Nk, V) -> (N, kV)
            expanded_score = tf.reshape(expanded_score, shape=[-1, self.width * n_classes])
        else:
            expanded_score = new_score  # shape (N, V)
        return tf.nn.top_k(expanded_score, k=self.width)

    def update(self, observation: tf.Tensor):
        n_classes = observation.shape[-1].value
        score, expanded_ids = self.select_beam(observation, n_classes)
        if self.score is not None:
            beam_ids = expanded_ids // n_classes  # shape (N, k), in range [0, k)
            class_ids = expanded_ids % n_classes  # shape (N, k) in range [0, V)
            self.flatten_ids = tf.reshape(
                self.offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
                shape=[-1],
            )  # shape (Nk, ): k elements in [0, k), k elements in [k, 2k)...
        else:
            class_ids = expanded_ids

        self.score = score
        if self.history is not None:
            self.history = self.select(self.history)

        observation = self.select(observation)
        class_ids = tf.reshape(class_ids, shape=[-1])
        return observation, class_ids

    def select(self, params):
        if self.flatten_ids is not None:
            return nested_call(tf.gather, params, indices=self.flatten_ids)
        return nested_call(tile_batch, params, multiplier=self.width)

    def get_top(self):
        output_ids = tf.squeeze(self.offset, axis=-1)  # shape (N, )
        return nested_call(tf.gather, self.history, indices=output_ids)


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
