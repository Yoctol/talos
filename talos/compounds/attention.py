import numpy as np
import tensorflow as tf

from talos.networks import Model


class ScaledDotSelfAttention(Model):

    def __init__(
            self,
            units: int,
            heads: int = 1,
            use_bias: bool = False,
        ):
        super().__init__()
        self.units = units
        self.heads = heads
        self.use_bias = use_bias

        self.query_layer = tf.keras.layers.Dense(units=units * heads)
        self.key_layer = tf.keras.layers.Dense(units=units * heads)
        self.value_layer = tf.keras.layers.Dense(units=units * heads)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
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
        # (N, h, T', T) * (N, h, T, U) -> (N, h, T', U)
        outputs = tf.matmul(weights, value, transpose_b=True)  # shape (N, h, T', U)

        outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])  # shape (N, T', h, U)
        outputs = tf.reshape(
            outputs, shape=[-1, width, self.heads * self.units])  # shape (N, T', hU)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.heads * self.units
        return tf.TensorShape(output_shape)
