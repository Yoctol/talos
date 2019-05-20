import abc

import tensorflow as tf

from .utils import apply_mask, ComputeOutputMaskMixin1D


class MaskAveragePooling1D(ComputeOutputMaskMixin1D, tf.keras.layers.AveragePooling1D):

    def call(self, inputs, mask=None):
        inputs = apply_mask(inputs, mask=mask)
        outputs = super().call(inputs)
        if mask is None:
            return outputs

        mask = tf.cast(mask, inputs.dtype)
        if self.data_format == 'channels_last':
            mask = mask[:, :, tf.newaxis, tf.newaxis]  # (N, W, H=1, C=1)
            h_axis = 2
        else:
            mask = mask[:, tf.newaxis, :, tf.newaxis]  # (N, C=1, W, H=1)
            h_axis = 3
        avg_true = self.pool_function(
            mask,
            self.pool_size + (1,),
            strides=self.strides + (1,),
            padding=self.padding,
            data_format=self.data_format,
        )
        avg_true = tf.squeeze(avg_true, h_axis)
        return outputs / (avg_true + tf.keras.backend.epsilon())


class MaskGlobalPooling1D(tf.keras.layers.Layer, abc.ABC):
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


class MaskGlobalAveragePooling1D(MaskGlobalPooling1D):

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)

        mask = tf.cast(mask, inputs.dtype)
        masked_inputs = apply_mask(inputs, mask)
        sum_inputs = tf.reduce_sum(masked_inputs, axis=1)  # shape (N, d_in)
        true_count = tf.reduce_sum(mask, axis=1, keepdims=True)  # shape (N, 1)
        return sum_inputs / (true_count + tf.keras.backend.epsilon())


# Aliases
MaskGlobalAvgPool1D = MaskGlobalAveragePooling1D
MaskAvgPool1D = MaskAveragePooling1D
