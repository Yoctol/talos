from typing import Tuple

import numpy as np
import tensorflow as tf

from talos.networks import Model


_LARGE_BIAS = 1e4


class _MultiHeadScaledDotAttention(Model):

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

    @property  # override property
    def input_spec(self):
        return self._input_spec

    def _get_glorot_uniform_initializer(self, input_shape):
        fan_in, fan_out = input_shape[-1].value, self.units
        limit = np.sqrt(6. / (fan_in + fan_out))  # glorot uniform
        return tf.keras.initializers.uniform(-limit, limit)

    def _multihead_attention(self, query, key, value, mask=None):
        if self.heads > 1:
            # shape (N, T, hU) -> (N, hU, T) -> (N, h, U, T)
            query, key, value = [
                tf.reshape(
                    tf.transpose(tensor, perm=[0, 2, 1]),
                    shape=[-1, self.heads, self.units, tensor.shape[1].value],
                )
                for tensor in (query, key, value)
            ]
            logits = tf.matmul(query, key, transpose_a=True)   # shape (N, h, T', T)
        else:
            logits = tf.matmul(query, key, transpose_b=True)  # shape (N, T', T)

        logits = self._mask_logits(logits / np.sqrt(self.units), mask)
        weights = tf.nn.softmax(logits)  # shape (N, h, T', T) or (N, T', T)

        if self.heads > 1:
            # (N, h, U, T) * (N, h, T, T') -> (N, h, U, T')
            attended_vec = tf.matmul(value, weights, transpose_b=True)
            attended_vec = tf.reshape(
                attended_vec,
                shape=[-1, self.heads * self.units, attended_vec.shape[-1].value],
            )  # shape (N, hU, T')
            attended_vec = tf.transpose(attended_vec, perm=[0, 2, 1])  # shape (N, T', hU)
        else:
            # (N, T', T) * (N, T, U) -> (N, T', U)
            attended_vec = tf.matmul(weights, value)

        return attended_vec

    def _mask_logits(self, logits, mask):
        if mask is None:
            return logits

        bias = (1. - mask) * _LARGE_BIAS
        if self.heads > 1:
            bias = bias[:, tf.newaxis, tf.newaxis, :]  # shape (N, 1, 1, T)
        else:
            bias = bias[:, tf.newaxis, :]  # shape (N, 1, T)

        return logits - bias

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask


class MultiHeadSelfAttention(_MultiHeadScaledDotAttention):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
            use_forward_mask: bool = False,
        ):
        super().__init__(
            units=units,
            output_dim=output_dim,
            heads=heads,
            activation=activation,
            use_bias=use_bias,
        )
        self.use_forward_mask = use_forward_mask
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        # Reference: https://tunz.kr/post/4
        # In order to use glorot uniform with fan_out = units instead of units * heads
        self.query_layer, self.key_layer, self.value_layer = [
            tf.keras.layers.Dense(
                name=name,
                units=self.units * self.heads,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self._get_glorot_uniform_initializer(input_shape),
            )
            for name in ('query_dense', 'key_dense', 'value_dense')
        ]
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)
            inputs *= mask[:, :, tf.newaxis]  # shape (N, T, d_in)

        query, key, value = [
            layer(inputs)
            for layer in (self.query_layer, self.key_layer, self.value_layer)
        ]  # shape (N, T, hU)

        attended_vec = self._multihead_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )
        outputs = self.output_layer(attended_vec)
        return outputs

    def _mask_logits(self, logits, mask):
        if self.use_forward_mask:
            width = logits.shape[-1].value
            logits -= tf.constant(
                np.triu(np.full([width, width], _LARGE_BIAS), k=1),
                dtype=logits.dtype,
            )
            # matrix of shape (T', T), Mt't = 1. if t' >= t else 0.
            # where t' come from query, t come from key, value.
            # [[0, 1e4, 1e4],
            #  [0, 0  , 1e4],
            #  [0, 0  ,   0]]

        logits = super()._mask_logits(logits, mask)
        return logits


class MultiHeadAttention(_MultiHeadScaledDotAttention):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
        ):
        super().__init__(
            units=units,
            output_dim=output_dim,
            heads=heads,
            activation=activation,
            use_bias=use_bias,
        )
        self._input_spec = [tf.keras.layers.InputSpec(ndim=3) for _ in range(2)]

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise TypeError("both 'inputs' should be length 2 tuple!")

        # Reference: https://tunz.kr/post/4
        # In order to use glorot uniform with fan_out = units instead of units * heads

        prev_input_shape, encoder_output_shape = input_shape
        self.query_layer, self.key_layer, self.value_layer = [
            tf.keras.layers.Dense(
                name=name,
                units=self.units * self.heads,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self._get_glorot_uniform_initializer(shape),
            )
            for name, shape in zip(
                ['query_dense', 'key_dense', 'value_dense'],
                [prev_input_shape, encoder_output_shape, encoder_output_shape],
            )
        ]
        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            mask: Tuple[tf.Tensor, tf.Tensor] = None,
        ) -> tf.Tensor:
        if not (len(inputs) == 2):
            raise TypeError("both 'inputs' and 'mask' should be length 2 tuple!")

        prev_inputs, encoder_outputs = inputs

        query = self.query_layer(prev_inputs)  # shape (N, T', hU)
        key, value = [
            layer(encoder_outputs)
            for layer in (self.key_layer, self.value_layer)
        ]  # shape (N, T, hU)

        attended_vec = self._multihead_attention(
            query=query,
            key=key,
            value=value,
        )
        outputs = self.output_layer(attended_vec)
        return outputs, encoder_outputs

    def compute_output_shape(self, input_shape):
        input_shape, encoder_outputs_shape = input_shape
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape), encoder_outputs_shape
