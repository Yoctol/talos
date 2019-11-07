import types
from typing import Set

import tensorflow as tf
from tensorflow.python.keras.layers.cudnn_recurrent import _CuDNNRNN

_WEIGHTS_VARIABLE_NAME = "kernel"


def add_spectral_norm(layer: tf.layers.Layer):
    if isinstance(layer, tf.keras.Sequential):
        for sub_layer in layer.layers:
            add_spectral_norm(sub_layer)
    elif isinstance(layer, tf.keras.Model):
        add_spectral_norm_for_model(layer)
    elif isinstance(layer, _CuDNNRNN):
        add_spectral_norm_for_layer(layer)  # unlike RNN, not a wrapper of cell
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
        lipschitz: float = 1.,
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

        spectral_norm, update_u = _build_spectral_norm_variables(name, kernel, original_add_weight)
        self.add_update(update_u)

        scale = lipschitz / (spectral_norm + tf.keras.backend.epsilon())
        normed_kernel = tf.multiply(kernel, scale, name=f'{name}_sn')
        return normed_kernel

    layer.add_weight = types.MethodType(new_add_weight, layer)


def _build_spectral_norm_variables(name, kernel, add_weight_func=tf.get_variable):
    kernel = to_rank2(kernel)  # shape (U, V)
    u_vector = add_weight_func(
        name=f'{name}/left_singular_vector',
        shape=(kernel.shape[0].value, ),
        initializer=tf.keras.initializers.lecun_normal(),  # unit vector
        trainable=False,
        dtype=kernel.dtype,
    )  # shape (U)

    new_v = tf.stop_gradient(
        tf.nn.l2_normalize(tf.linalg.matvec(kernel, u_vector, transpose_a=True)),
        name=f'{name}/new_right_singular_vector',
    )  # shape (V)
    Wv = tf.linalg.matvec(kernel, new_v)  # shape (U)
    new_u = tf.stop_gradient(
        tf.nn.l2_normalize(Wv),
        name=f'{name}/new_left_singular_vector',
    )  # shape (U)
    spectral_norm = tf.tensordot(new_u, Wv, axes=1, name=f'{name}/singular_value')
    update_u = tf.assign(u_vector, new_u, name=f'{name}/power_iter')
    return spectral_norm, update_u


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
