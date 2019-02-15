import abc

import tensorflow as tf


class GlobalPooling1D(tf.keras.layers.Layer, abc.ABC):
    """Abstract class for different global pooling 1D layers.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], input_shape[2]])

    def compute_mask(self, inputs, mask):
        return None

    @abc.abstractmethod
    def call(self, inputs, mask=None):
        pass


class GlobalAveragePooling1D(GlobalPooling1D):

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)

        mask = tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]  # shape (N, T, 1)
        masked_inputs = tf.reduce_sum(inputs * mask, axis=1)  # shape (N, d_in)
        true_count = tf.reduce_sum(mask, axis=1)  # shape (N, 1)
        return masked_inputs / (true_count + tf.keras.backend.epsilon())


# Aliases
GlobalAvgPool1D = GlobalAveragePooling1D
