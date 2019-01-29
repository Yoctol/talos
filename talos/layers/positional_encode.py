import numpy as np
import tensorflow as tf


class PositionalEncode(tf.keras.layers.Layer):

    def __init__(
            self,
            base: float = 1e4,
            amplitude: float = 1.,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.base = base
        self.amplitude = amplitude
        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        dtype = inputs.dtype
        maxlen, dim = inputs.shape.as_list()[1:]
        pe = self._get_positional_encode_tensor(maxlen, dim, dtype)
        return inputs + pe[tf.newaxis, :, :]

    def _get_positional_encode_tensor(self, maxlen, dim, dtype):
        position_range = np.arange(maxlen)  # shape [L]
        dim_range = np.arange(dim)  # shape [D]
        wave_length = np.power(self.base, 2. * dim_range / dim)  # shape [D]

        offset = -np.pi * ((dim_range % 2) + 1)
        # [-pi / 2, 0, ...] for convert sin to cos on even dim, shape [D]

        theta = position_range[:, np.newaxis] / wave_length[np.newaxis, :] + offset[np.newaxis, :]
        outputs_np = self.amplitude * np.cos(theta)
        return tf.constant(outputs_np, dtype=dtype)  # shape [L, D]

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        return mask
