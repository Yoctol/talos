import types
from typing import Set

import tensorflow as tf

_WEIGHTS_VARIABLE_NAME = "kernel"


def add_spectral_norm(
        layer: tf.layers.Layer,
        kernel_name: Set[str] = None,
    ):
    if layer.built:
        raise ValueError("Can't add spectral norm on built layer!")

    def is_kernel_var(name):
        if kernel_name is not None:
            return name in kernel_name
        else:
            return name.endswith(_WEIGHTS_VARIABLE_NAME)

    original_add_weight = layer.add_weight

    # Very Very Evil HACK
    def new_add_weight(self, name=None, shape=None, **kwargs):
        kernel = original_add_weight(name, shape, **kwargs)
        if not is_kernel_var(name):
            return kernel

        if len(shape) < 2:
            raise ValueError("Can't apply spectral norm on variable rank < 2!")

        kernel_matrix = to_rank2(kernel)  # shape (U, V)

        u_vector = original_add_weight(
            name=f'{name}/singular_vector',
            shape=(kernel_matrix.shape[0].value, 1),
            initializer=tf.keras.initializers.lecun_normal(),  # unit vector
            trainable=False,
            dtype=kernel.dtype,
        )  # shape (U, 1)
        v_vector = tf.matmul(kernel_matrix, u_vector, transpose_a=True)  # shape (V, 1)
        spectral_norm = tf.norm(v_vector)  # shape (1)
        normed_kernel = tf.truediv(
            kernel,
            spectral_norm + tf.keras.backend.epsilon(),
            name=f'{name}_sn',
        )

        new_u = tf.nn.l2_normalize(kernel_matrix @ v_vector, axis=0)  # shape (U, 1)
        update_u = tf.assign(u_vector, new_u, name=f'{name}_sn/power_iter')
        self.add_update(update_u)

        return normed_kernel

    layer.add_weight = types.MethodType(new_add_weight, layer)


def to_rank2(tensor: tf.Tensor):
    if tensor.shape.ndims > 2:
        return tf.reshape(tensor, [-1, tensor.shape[-1].value])
    return tensor
