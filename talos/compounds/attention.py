import numpy as np
import tensorflow as tf

from talos.networks import Model


class ScaledDotSelfAttention(Model):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
        ):
        super().__init__()
        self.units = units
        self.heads = heads
        self.output_dim = output_dim

        self.supports_masking = True

        self.query_layer, self.key_layer, self.value_layer = [
            tf.keras.layers.Dense(
                name=name,
                units=units * heads,
                activation=activation,
                use_bias=use_bias,
            )
            for name in ('query_dense', 'key_dense', 'value_dense')
        ]
        self.output_layer = tf.keras.layers.Dense(
            name='output_dense',
            units=output_dim,
            use_bias=use_bias,
        )  # just for parametrization
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property  # override property
    def input_spec(self):
        return self._input_spec

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)
            inputs *= mask[:, :, tf.newaxis]  # shape (N, T, d_in)

        query = self.query_layer(inputs)  # shape (N, T', hU)
        key = self.key_layer(inputs)  # shape (N, T, hU)
        value = self.value_layer(inputs)  # shape (N, T, hU)

        width = inputs.shape[1].value
        matmul_broadcast_shape = [-1, self.heads, self.units, width]
        # shape (N, T, hU) -> (N, hU, T) -> (N, h, U, T)
        query = tf.reshape(tf.transpose(query, perm=[0, 2, 1]), shape=matmul_broadcast_shape)
        key = tf.reshape(tf.transpose(key, perm=[0, 2, 1]), shape=matmul_broadcast_shape)
        value = tf.reshape(tf.transpose(value, perm=[0, 2, 1]), shape=matmul_broadcast_shape)

        logits = tf.matmul(
            query, key, transpose_a=True) / np.sqrt(self.units)  # shape (N, h, T', T)
        weights = tf.nn.softmax(logits)  # shape (N, h, T', T)

        if mask is not None:
            # Renormalize for lower seqlen
            weights *= mask[:, tf.newaxis, tf.newaxis, :]  # shape (N, 1, 1, T)
            weights /= (tf.reduce_sum(weights, axis=3, keepdims=True) + tf.keras.backend.epsilon())

        # (N, h, T', T) * (N, h, T, U) -> (N, h, T', U)
        attended_vec = tf.matmul(weights, value, transpose_b=True)  # shape (N, h, T', U)

        attended_vec = tf.transpose(attended_vec, perm=[0, 2, 1, 3])  # shape (N, T', h, U)
        attended_vec = tf.reshape(
            attended_vec, shape=[-1, width, self.heads * self.units])  # shape (N, T', hU)
        outputs = self.output_layer(attended_vec)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
