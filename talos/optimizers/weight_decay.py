from typing import Callable, Container, Union

import tensorflow as tf


class WeightDecay(tf.train.Optimizer):

    '''Reference: https://arxiv.org/pdf/1711.05101.pdf'''

    def __init__(
            self,
            optimizer,
            decay_rate: float,
            use_locking: bool = False,
            name: str = 'WeightDecay',
            variable_filter: Union[Container[tf.Variable], Callable[[tf.Variable], bool]] = None,
        ):
        super().__init__(use_locking, name)
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_rate_tensor = tf.convert_to_tensor(decay_rate)
        self.variable_filter = variable_filter

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        var_list, decay_value = self._get_decay_pairs(grads_and_vars)
        with tf.control_dependencies(decay_value):  # cache the value before descent
            grad_descent_op = self.optimizer.apply_gradients(
                grads_and_vars,
                global_step=global_step,
            )

        with tf.control_dependencies([grad_descent_op]):  # guarantee compute before decay.
            decay_op = tf.group(
                *[
                    v.assign_sub(d_v, use_locking=self._use_locking)
                    for v, d_v in zip(var_list, decay_value)
                ],
                name=name,
            )

        return decay_op

    def _get_decay_pairs(self, grads_and_vars):
        if self.variable_filter is None:
            def need_decay(var):
                return True
        elif hasattr(self.variable_filter, '__contains__'):
            def need_decay(var):
                return var in self.variable_filter
        else:
            need_decay = self.variable_filter

        var_list = [v for g, v in grads_and_vars if g is not None and need_decay(v)]
        decay_value = [
            tf.cast(self.decay_rate_tensor, dtype=v.dtype.base_dtype) * v
            for v in var_list
        ]
        return var_list, decay_value
