import types
from typing import Set

import tensorflow as tf

_WEIGHTS_VARIABLE_NAME = "kernel"


def add_spectral_norm(
        layer: tf.layers.Layer,
        kernel_name: Set[str] = None,
        eps: float = 1e-7,
    ):
    # Very Very Evil HACK
    assert not layer.built, "Can't add spectral norm on built layer!"

    def is_kernel_var(name):
        if kernel_name is not None:
            return name in kernel_name
        else:
            return name.endswith(_WEIGHTS_VARIABLE_NAME)

    original_add_weight = layer.add_weight

    def new_add_weight(self, name, shape, **kwargs):
        kernel = original_add_weight(name, shape, **kwargs)
        if is_kernel_var(name):
            assert len(shape) >= 2, "Can't apply spectral norm on variable rank < 2!"
            if len(shape) > 2:
                kernel_matrix = tf.reshape(kernel, [-1, kernel.shape[-1]])
            else:
                kernel_matrix = kernel

            u_vector = original_add_weight(
                name=f'{name}/singular_vector',
                shape=(kernel_matrix.shape[0].value, 1),
                initializer=tf.truncated_normal_initializer(),
                trainable=False,
                dtype=kernel.dtype,
            )
            v_vector = tf.matmul(kernel_matrix, u_vector, transpose_a=True)
            spectral_norm = tf.norm(v_vector, axis=0)
            new_u = tf.nn.l2_normalize(kernel_matrix @ v_vector, axis=0)
            update_u = tf.assign(
                u_vector, new_u, name=f'{name}/power_iter',
            )
            self.add_update(update_u)

            kernel = tf.truediv(kernel, spectral_norm + eps, name=f'{name}_sn')

        return kernel

    layer.add_weight = types.MethodType(new_add_weight, layer)
