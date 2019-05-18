import abc

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from .utils import apply_mask


class MaskAveragePooling1D(tf.keras.layers.AveragePooling1D):

    def __init__(
            self,
            pool_size=2,
            strides=None,
            padding='valid',
            data_format='channels_last',
            mask_threshold=None,
            **kwargs,
        ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs,
        )
        if not (mask_threshold is None or 1 <= mask_threshold <= pool_size):
            raise ValueError(f"`mask_threshold` should be in [1, {pool_size}]")
        self.mask_threshold = mask_threshold
        self.supports_masking = True

    def build(self, input_shape):
        super().build(input_shape)
        self.mask_kernel = tf.ones(self.pool_size + (1, 1), dtype=self.dtype)
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding

        mask_shape = (*input_shape.as_list()[:-1], 1)
        self._mask_op = nn_ops.Convolution(
            tf.TensorShape(mask_shape),
            filter_shape=self.mask_kernel.get_shape(),
            strides=self.strides,
            padding=op_padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 3),
        )

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
        avg_true = array_ops.squeeze(avg_true, h_axis)
        return outputs / (avg_true + tf.keras.backend.epsilon())

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        mask = tf.expand_dims(tf.cast(mask, self.dtype), axis=channel_axis)

        output_mask = self._mask_op(mask, self.mask_kernel)  # float

        output_mask = tf.squeeze(output_mask, axis=channel_axis)

        if self.mask_threshold is not None:
            mask_threshold = self.mask_threshold
        elif self.padding == 'same':
            mask_threshold = self.pool_size[0] / 2
        else:
            mask_threshold = self.pool_size[0]

        return tf.greater(output_mask, mask_threshold - 0.1)  # avoid rounding error


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
