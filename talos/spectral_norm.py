import types
from typing import Set

import tensorflow as tf

_WEIGHTS_VARIABLE_NAME = "kernel"


def add_spectral_norm(layer: tf.layers.Layer):
    if isinstance(layer, tf.keras.Sequential):
        for sub_layer in layer.layers:
            add_spectral_norm(sub_layer)
    elif isinstance(layer, tf.keras.Model):
        add_spectral_norm_for_model(layer)
    elif isinstance(layer, tf.keras.layers.RNN):
        add_spectral_norm(layer.cell)
    elif isinstance(layer, tf.keras.layers.StackedRNNCells):
        for cell in layer.cells:
            add_spectral_norm(cell)
    elif isinstance(layer, tf.keras.layers.Bidirectional):
        add_spectral_norm(layer.forward_layer)
        add_spectral_norm(layer.backward_layer)
    elif isinstance(layer, tf.keras.layers.Wrapper):
        add_spectral_norm(layer.layer)
    else:
        add_spectral_norm_for_layer(layer)


def add_spectral_norm_for_layer(
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
            name=f'{name}/left_singular_vector',
            shape=(kernel_matrix.shape[0].value, 1),
            initializer=tf.keras.initializers.lecun_normal(),  # unit vector
            trainable=False,
            dtype=kernel.dtype,
        )  # shape (U, 1)

        new_v = tf.stop_gradient(
            tf.nn.l2_normalize(tf.matmul(kernel_matrix, u_vector, transpose_a=True)),
            name=f'{name}/new_right_singular_vector',
        )  # shape (V, 1)
        unnormed_new_u = kernel_matrix @ new_v  # shape (U, 1)
        new_u = tf.stop_gradient(
            tf.nn.l2_normalize(unnormed_new_u),
            name=f'{name}/new_left_singular_vector',
        )
        spectral_norm = tf.squeeze(
            tf.matmul(new_u, unnormed_new_u, transpose_a=True),
            name=f'{name}/singular_value',
        )
        normed_kernel = tf.truediv(
            kernel,
            spectral_norm + tf.keras.backend.epsilon(),
            name=f'{name}_sn',
        )
        update_u = tf.assign(u_vector, new_u, name=f'{name}/power_iter')
        self.add_update(update_u)

        return normed_kernel

    layer.add_weight = types.MethodType(new_add_weight, layer)


def add_spectral_norm_for_model(model: tf.keras.Model):
    if model.built:
        raise ValueError("Can't add spectral norm on built layer!")

    original_build = model.build

    # Very Very Evil HACK
    def new_build(self, input_shape):
        original_build(input_shape)
        for sub_layer in self.layers:
            add_spectral_norm(sub_layer)

    model.build = types.MethodType(new_build, model)


def to_rank2(tensor: tf.Tensor):
    if tensor.shape.ndims > 2:
        return tf.reshape(tensor, [-1, tensor.shape[-1].value])
    return tensor
