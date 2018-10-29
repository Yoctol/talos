import types

import tensorflow as tf

_WEIGHTS_VARIABLE_NAME = "kernel"


def get_sn_kernel(
        layer,
        kernel: tf.Variable,
        eps: float = 1e-8,
    ) -> tf.Variable:
    rank = len(kernel.shape.as_list())
    if rank > 2:
        kernel_matrix = tf.reshape(kernel, [-1, kernel.shape[-1]])
    else:
        kernel_matrix = kernel

    u_vector = layer.add_variable(
        name=f'{kernel.name[::-2]}/u_vector',
        shape=(kernel_matrix.shape[0], 1),
        initializer=tf.truncated_normal_initializer(),
        trainable=False,
        dtype=layer.dtype,
    )
    v_vector = tf.matmul(kernel_matrix, u_vector, transpose_a=True)
    spectral_norm = tf.norm(v_vector, axis=0)
    new_u = tf.nn.l2_normalize(kernel_matrix @ v_vector, axis=0)
    update_u = tf.assign(
        u_vector, new_u, name=f'{kernel.name[::-2]}/power_iter',
    )
    layer.add_update(update_u)

    sn_kernel = tf.truediv(kernel, spectral_norm + eps, name=f'{kernel.name[::-2]}_sn')
    return sn_kernel


def add_spectral_norm(layer: tf.layers.Layer):
    assert not layer.built
    original_build = layer.build

    def new_build(self, input_shapes):
        original_build(input_shapes)
        self.built = False
        kernels = {
            key: get_sn_kernel(self, val)
            for key, val in self.__dict__.items() if key.endswith('kernel')
        }
        self.__dict__.update(kernels)
        self.built = True

    layer.build = types.MethodType(new_build, layer)
