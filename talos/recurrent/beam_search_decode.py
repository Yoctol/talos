from typing import Callable, List

import tensorflow as tf

from .chain_decode import RECURRENT_CELL_TYPE, get_default_init_state


def beam_search_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        beam_width: int,
        next_input_producer: Callable,
        end_token: int,
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
    beam = Beam(batch_size=batch_size, width=beam_width, end_token=end_token)

    for _ in range(maxlen):
        output, state = cell(inputs, state)  # shape (Nk, d_o), (Nk, d_s)
        choice_ids = beam.explore_step(observation=output)  # 更新 beam
        # NOTE 以下兩行一定要call，且順序不可改變。
        output, state = beam.select([output, state])  # 幫我 tf.gather 這個東西
        beam.append([output, choice_ids])  # 加進 beam history 中
        # NOTE
        inputs = next_input_producer(output, choice_ids)  # shape (N, d_i)

    output_tensor, output_choice_ids = beam.get_top()

    return output_tensor, output_choice_ids


class Beam:

    _TIME_AXIS = 1

    def __init__(
            self,
            batch_size,
            width,
            end_token,
            score_func=tf.nn.log_softmax,
        ):
        self.width = width
        self.end_token = end_token

        self._history = None
        self._sum_score = None
        self._flatten_ids = None
        self._seqlen = tf.ones([batch_size, 1])
        self._is_not_finished = None
        self._offset = tf.expand_dims(
            tf.range(batch_size) * width,
            axis=1,
        )  # shape (N, 1): 0, k, 2k....
        self._score_func = score_func
        self._n_steps = 0

    def append(self, tensors: List):
        tensors = [tf.expand_dims(t, axis=self._TIME_AXIS) for t in tensors]
        if self._history is not None:
            assert len(tensors) == len(self._history)
            assert all(
                h.shape[self._TIME_AXIS].value == self._n_steps - 1
                for h in self._history
            ), "you should call `append` after every `update`!"
            self._history = self.select(self._history)
            self._history = [
                tf.concat([h, t], axis=self._TIME_AXIS)
                for h, t in zip(self._history, tensors)
            ]
        else:
            assert self._n_steps == 1, "you should call `append` after every `update`!"
            self._history = tensors

    def explore_step(self, observation: tf.Tensor):
        n_choices = observation.shape[-1].value
        top_k_ave_score, expanded_ids = self._expand_beam(observation, n_choices)
        # expand_ids in range [0, kV)
        if self._n_steps > 0:
            beam_ids = expanded_ids // n_choices  # shape (N, k), in range [0, k)
            choice_ids = expanded_ids % n_choices  # shape (N, k) in range [0, V)
            self._flatten_ids = tf.reshape(
                self._offset + beam_ids,  # shape (N, 1) + (N, k) -> (N, k)
                shape=[-1],
            )  # shape (Nk, ): k elements in [0, k), k elements in [k, 2k)...
        else:
            choice_ids = expanded_ids

        self._n_steps += 1
        choice_ids = tf.reshape(choice_ids, shape=[-1, 1])  # shape (Nk, 1)
        if self._is_not_finished is not None:
            self._is_not_finished = tf.logical_and(
                self._is_not_finished,
                tf.not_equal(choice_ids, self.end_token),
            )
        else:
            self._is_not_finished = tf.not_equal(choice_ids, self.end_token)
        self._seqlen = self.select(self._seqlen)
        self._seqlen += tf.cast(self._is_not_finished, tf.float32)
        self._sum_score = tf.reshape(top_k_ave_score, shape=[-1, 1]) * self._seqlen
        return tf.squeeze(choice_ids, axis=1)

    def _expand_beam(self, observation, n_choices):
        new_score = self._score_func(observation)  # shape (Nk, V)
        if self._is_not_finished is not None:
            new_score *= tf.cast(self._is_not_finished, tf.float32)
        if self._n_steps > 0:
            # broadcast: (Nk, 1) + (Nk, V) -> (Nk, V)
            expanded_score = self._sum_score + new_score
            ave_score = expanded_score / self._seqlen
            ave_score = tf.reshape(
                ave_score, shape=[-1, self.width * n_choices]
            )  # shape (N, kV)
        else:
            ave_score = new_score  # shape (N, V)
        return tf.nn.top_k(ave_score, k=self.width)

    def select(self, params):
        if self._flatten_ids is not None:
            return nested_call(tf.gather, params, indices=self._flatten_ids)
        return nested_call(tile_batch, params, multiplier=self.width)

    def get_top(self):
        output_ids = tf.squeeze(self._offset, axis=-1)  # shape (N, )
        return nested_call(tf.gather, self._history, indices=output_ids)


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
