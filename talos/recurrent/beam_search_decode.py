from typing import Callable, List, Sequence, Tuple, Union

import tensorflow as tf

from .chain_decode import RECURRENT_CELL_TYPE, get_default_init_state


NESTED_TYPE = Union[Sequence, tf.Tensor]


def beam_search_decode(
        cell: RECURRENT_CELL_TYPE,
        first_input: tf.Tensor,
        maxlen: int,
        beam_width: int,
        next_input_producer: Callable,
        end_token: int,
        init_state: tf.Tensor = None,
        output_width: int = 1,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    # example of next_input_producer: lambda logit: embedding_lookup(argmax(logit))
    if init_state is None:
        init_state = get_default_init_state(cell, first_input)

    inputs = first_input  # shape (N, d_i)
    state = init_state  # shape (N, d_s)
    beam = Beam(width=beam_width, end_token=end_token)

    for _ in range(maxlen):
        output, state = cell(inputs, state)  # shape (N, k, d_o), (N, k, d_s)
        choice_ids = beam.explore_step(observation=output)  # 更新 beam
        # NOTE 以下兩行一定要call，且順序不可改變。
        output, state = beam.select([output, state])  # 幫我 tf.batch_gather 這個東西
        beam.append([output, choice_ids])  # 加進 beam history 中
        # NOTE
        inputs = next_input_producer(output, choice_ids)  # shape (N, k, d_i)

    output_tensor, output_choice_ids = beam.get_top(output_width=output_width)

    return output_tensor, output_choice_ids


def _compute_length_penalty(func):
    def wrapper(self, score):
        if self._seqlen is not None:
            length_penalty_coeff = self._length_penalty(self._seqlen)
            ave_score, expanded_ids = func(self, score / length_penalty_coeff)
            sum_score = ave_score * tf.squeeze(length_penalty_coeff, axis=2)
            return sum_score, expanded_ids
        return func(self, score)
    return wrapper


class Beam:

    _TIME_AXIS = 2

    def __init__(
            self,
            width: int,
            score_func: Callable = tf.nn.log_softmax,
            end_token: int = None,
            length_penalty: Callable = lambda x: x,
        ):
        self.width = width
        self.end_token = end_token

        self._history = None
        self._sum_score = None
        self._beam_ids = None

        self._length_penalty = length_penalty
        self._not_finished = None
        self._seqlen = None

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
        self._sum_score, expanded_ids = self._select_new_beam(observation)
        if observation.shape.ndims == 3:
            # expand_ids in range [0, kV)
            n_choices = observation.shape[-1].value
            self._beam_ids = expanded_ids // n_choices  # shape (N, k), in range [0, k)
            choice_ids = expanded_ids % n_choices  # shape (N, k) in range [0, V)
        else:
            # expand_ids in range [0, V)
            choice_ids = expanded_ids  # shape (N, k)

        if self.end_token is not None:
            self._seqlen, self._not_finished = self._new_seqlen_finished(choice_ids)
        self._n_steps += 1
        return choice_ids

    def _select_new_beam(self, observation):
        step_score = self._score_func(observation)  # shape (N, k, V)
        if self._not_finished is not None:
            step_score *= tf.to_float(self._not_finished)
        if self._sum_score is not None:
            # broadcast: (N, k, 1) + (N, k, V) -> (N, k, V)
            sum_score = tf.expand_dims(self._sum_score, axis=2) + step_score
        else:
            sum_score = step_score
        return self._top_k_of_beam_and_choices(sum_score)

    @_compute_length_penalty
    def _top_k_of_beam_and_choices(self, score):
        if score.shape.ndims == 3:
            last_dim = score.shape[1].value * score.shape[2].value
            score = tf.reshape(score, shape=[-1, last_dim])
        return tf.nn.top_k(score, k=self.width)

    def _new_seqlen_finished(self, choice_ids):
        # every tensor have shape (N, k, 1)
        step_not_finished = tf.not_equal(choice_ids, self.end_token)
        step_not_finished = tf.expand_dims(step_not_finished, axis=2)
        if self._not_finished is not None:
            new_not_finished = tf.logical_and(self._not_finished, step_not_finished)
        else:
            new_not_finished = step_not_finished

        length_to_add = tf.to_float(new_not_finished)
        prev_seqlen = self.select(self._seqlen) if self._seqlen is not None else 1.
        new_seqlen = length_to_add + prev_seqlen
        return new_seqlen, new_not_finished

    def select(self, params: NESTED_TYPE) -> NESTED_TYPE:
        if self._beam_ids is not None:
            return nested_call(tf.batch_gather, params, indices=self._beam_ids)
        return nested_call(tile_beam, params, multiplier=self.width)

    def get_top(self, output_width: int = 1):
        assert 0 < output_width <= self.width
        return [tensor[:, 0: output_width] for tensor in self._history]


def nested_call(
        func: Callable,
        nested_var: NESTED_TYPE,
        *args,
        **kwargs
    ) -> NESTED_TYPE:
    if hasattr(nested_var, '__len__'):
        return [nested_call(func, v, *args, **kwargs) for v in nested_var]
    return func(nested_var, *args, **kwargs)


def tile_beam(tensor: tf.Tensor, multiplier: int) -> tf.Tensor:
    shape = tensor.shape
    rank = shape.ndims
    assert rank >= 1
    tensor = tf.expand_dims(tensor, axis=1)
    tensor = tf.tile(tensor, multiples=[1, multiplier] + [1] * (rank - 1))
    return tensor
