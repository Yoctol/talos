import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class MaskConv1D(tf.keras.layers.Conv1D):

    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            mask_threshold=None,
            **kwargs,
        ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        if not (mask_threshold is None or 1 <= mask_threshold <= kernel_size):
            raise ValueError(f"`mask_threshold` should be in [1, {kernel_size}]")
        self.mask_threshold = mask_threshold
        self.supports_masking = True

    def build(self, input_shape):
        super().build(input_shape)
        self.mask_kernel = tf.ones(self.kernel_size + (1, 1), dtype=self.dtype)
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding

        mask_shape = (*input_shape.as_list()[:-1], 1)
        self._mask_op = nn_ops.Convolution(
            tf.TensorShape(mask_shape),
            filter_shape=self.mask_kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2),
        )

    def call(self, inputs, mask=None):
        outputs = super().call(inputs)
        return outputs

    def compute_mask(self, inputs, mask):
        if mask is None:
            return None

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        mask = tf.expand_dims(tf.cast(mask, self.dtype), axis=channel_axis)

        if self.padding == 'causal':
            mask = array_ops.pad(mask, self._compute_causal_padding())
        output_mask = self._mask_op(mask, self.mask_kernel)  # float

        output_mask = tf.squeeze(output_mask, axis=channel_axis)

        if self.mask_threshold is not None:
            mask_threshold = self.mask_threshold
        elif self.padding == 'same':
            mask_threshold = self.kernel_size[0] / 2
        else:
            mask_threshold = self.kernel_size[0]

        return tf.greater(output_mask, mask_threshold - 0.1)  # avoid rounding error
