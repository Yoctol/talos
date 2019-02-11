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
        self.output_dim = output_dim
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias

        self.supports_masking = True
        self.output_layer = tf.keras.layers.Dense(
            name='output_dense',
            units=self.output_dim,
            use_bias=self.use_bias,
        )  # just for parametrization
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property  # override property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        # Reference: https://tunz.kr/post/4
        # In order to use glorot uniform with fan_out = units instead of units * heads
        fan_in, fan_out = input_shape[-1].value, self.units
        limit = np.sqrt(6. / (fan_in + fan_out))  # glorot uniform
        self.query_layer, self.key_layer, self.value_layer = [
            tf.keras.layers.Dense(
                name=name,
                units=self.units * self.heads,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=tf.keras.initializers.uniform(-limit, limit),
            )
            for name in ('query_dense', 'key_dense', 'value_dense')
        ]
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)
            inputs *= mask[:, :, tf.newaxis]  # shape (N, T, d_in)

        width = inputs.shape[1].value
        # shape (N, T, hU) -> (N, hU, T) -> (N, h, U, T)
        if self.heads > 1:
            query, key, value = [
                tf.reshape(
                    tf.transpose(layer(inputs), perm=[0, 2, 1]),
                    shape=[-1, self.heads, self.units, width],
                )
                for layer in (self.query_layer, self.key_layer, self.value_layer)
            ]
            logits = tf.matmul(query, key, transpose_a=True)   # shape (N, h, T', T)
        else:
            query, key, value = [
                layer(inputs)
                for layer in (self.query_layer, self.key_layer, self.value_layer)
            ]  # shape (N, T, U)
            logits = tf.matmul(query, key, transpose_b=True)  # shape (N, T', T)

        weights = tf.nn.softmax(logits / np.sqrt(self.units))  # shape (N, h, T', T) or (N, T', T)

        if mask is not None:
            # Renormalize for lower seqlen
            if self.heads > 1:
                weights *= mask[:, tf.newaxis, tf.newaxis, :]  # shape (N, 1, 1, T)
            else:
                weights *= mask[:, tf.newaxis, :]  # shape (N, 1, T)
            weights /= (tf.reduce_sum(weights, axis=-1, keepdims=True) + tf.keras.backend.epsilon())

        if self.heads > 1:
            # (N, h, U, T) * (N, h, T, T') -> (N, h, U, T')
            attended_vec = tf.matmul(value, weights, transpose_b=True)
            attended_vec = tf.reshape(
                attended_vec, shape=[-1, self.heads * self.units, width])  # shape (N, hU, T')
            attended_vec = tf.transpose(attended_vec, perm=[0, 2, 1])  # shape (N, T', hU)
        else:
            # (N, T', T) * (N, T, U) -> (N, T', U)
            attended_vec = tf.matmul(weights, value)

        outputs = self.output_layer(attended_vec)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
