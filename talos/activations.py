import tensorflow as tf


ACTIVATIONS = {
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
