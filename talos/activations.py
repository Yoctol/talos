import numpy as np
import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.

    """
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


def gelu_v2(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.

    """
    return x * tf.nn.sigmoid(1.702 * x)


# NOTE register functions to keras, then we can get them by string key
# see the test

local_functions = {
    f"talos.{key}": val
    for key, val in locals().items()
    if callable(val) and val.__module__ == __name__
}

custom_object = tf.keras.utils.get_custom_objects()
for key in local_functions.keys():
    assert key not in custom_object, f"Name conflict! {key} has been defined alreay!"

custom_object.update(local_functions)
