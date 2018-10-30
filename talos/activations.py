import tensorflow as tf


_ACTIVATIONS = {
    'relu': tf.nn.relu,
    'lrelu': tf.nn.leaky_relu,
    'elu': tf.nn.elu,
    'selu': tf.nn.selu,
    'tanh': tf.nn.tanh,
    'sigmoid': tf.nn.sigmoid,
    'softmax': tf.nn.softmax,
    'linear': lambda x: x,
    None: None,
}


def get(activation_id: str):
    try:
        activation = _ACTIVATIONS[activation_id]
        return activation
    except KeyError:
        raise KeyError(f"Unknown activation_id: {activation_id}")
