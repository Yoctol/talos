from typing import Callable, Container, Union

import tensorflow as tf


class SpectralWeightDecay(tf.train.Optimizer):
    '''
    References:
        1. Decouple Weight Decay https://arxiv.org/abs/1711.05101
        2. Spectral Regularization https://arxiv.org/abs/1705.10941
    '''

    def __init__(
            self,
            optimizer,
            decay_rate: float,
            use_locking: bool = False,
            name: str = 'SpectralWeightDecay',
            variable_filter: Union[Container[tf.Variable], Callable[[tf.Variable], bool]] = None,
        ):
        super().__init__(use_locking, name)
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_rate_tensor = tf.convert_to_tensor(decay_rate)
        self.variable_filter = variable_filter

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        var_list, decay_value, update_list = self._get_decay_trips(grads_and_vars)
        with tf.control_dependencies(decay_value):  # cache the value before descent
            grad_descent_op = self.optimizer.apply_gradients(
                grads_and_vars,
                global_step=global_step,
            )

        # guarantee compute before decay.
        with tf.control_dependencies([grad_descent_op]):
            decay_op = tf.group(
                *[
                    v.assign_sub(d_v, use_locking=self._use_locking)
                    for v, d_v in zip(var_list, decay_value)
                ],
                *update_list,
                name=name,
            )

        return decay_op

    def _get_decay_trips(self, grads_and_vars):
        if self.variable_filter is None:
            def need_decay(var):
                return 'kernel' in v.name and v.shape.ndims >= 2
        elif hasattr(self.variable_filter, '__contains__'):
            def need_decay(var):
                return var in self.variable_filter
        else:
            need_decay = self.variable_filter

        var_list, decay_list, update_list = [], [], []
        for g, v in grads_and_vars:
            if g is None or not need_decay(v):
                continue
            if v.shape.ndims < 2:
                raise ValueError("Can't apply spectral norm on variable with rank < 2!")
            decay_value, update_u = self._build_spectral_norm_variables(v)
            rate = tf.cast(self.decay_rate_tensor, dtype=v.dtype.base_dtype)
            var_list.append(v)
            decay_list.append(rate * decay_value)
            update_list.append(update_u)

        return var_list, decay_list, update_list

    def _build_spectral_norm_variables(self, kernel):
        kernel_matrix = to_rank2(kernel)  # shape (U, V)
        u = self._get_or_make_slot_with_initializer(
            kernel,
            initializer=tf.keras.initializers.lecun_normal(),  # unit vector
            shape=kernel_matrix.shape[:1],
            dtype=kernel_matrix.dtype,
            slot_name='u',
            op_name=self._name,
        )  # shape (U)
        v = tf.nn.l2_normalize(tf.linalg.matvec(kernel_matrix, u, transpose_a=True))  # shape (V)
        Wv = tf.linalg.matvec(kernel_matrix, v)  # shape (U)
        # NOTE
        # sigma = u^T W v -> dsigma / dW = uv^T
        # 0.5 dsigma^2 / dW = sigma u v^T = (sigma u) v^T = Wv v^T
        decay_value = Wv[:, tf.newaxis] * v  # shape (U, V)
        if kernel.shape.ndims > 2:
            decay_value = tf.reshape(decay_value, kernel.shape)

        new_u = tf.nn.l2_normalize(Wv)  # shape (U)
        update_u = tf.assign(u, new_u)
        return decay_value, update_u


def to_rank2(tensor: tf.Tensor):
    if tensor.shape.ndims > 2:
        return tf.reshape(tensor, [-1, tensor.shape[-1].value])
    return tensor
