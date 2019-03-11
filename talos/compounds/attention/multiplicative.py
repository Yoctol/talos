from typing import List

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
            heads_reg_coeff: float = None,
        ):
        super().__init__()
        self.units = units
        self.output_dim = output_dim
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias

        if heads_reg_coeff is not None and heads_reg_coeff < 0:
            raise ValueError("reg_coeff can't be negative!")
        self.heads_reg_coeff = heads_reg_coeff

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

    def _multihead_attention(self, query, key, value, value_mask=None):
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

        logits = self._mask_logits(logits / np.sqrt(self.units), value_mask)
        weights = tf.nn.softmax(logits)  # shape (N, h, T', T) or (N, T', T)

        if self.heads > 1:
            # (N, h, U, T) * (N, h, T, T') -> (N, h, U, T')
            attended_vec = tf.matmul(value, weights, transpose_b=True)
            if self.heads_reg_coeff is not None:
                # NOTE add input loss more than once
                # may cause dependencies error
                if self.inputs:
                    raise RuntimeError(
                        "Layer with inputs related regularization "
                        "should not be called more than once!",
                    )
                reg_loss = self._disagreement_output_loss(attended_vec)
                self.output_layer.add_loss(self.heads_reg_coeff * reg_loss)

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

    def _disagreement_output_loss(self, attended_vec):
        width = attended_vec.shape[3].value
        unit_head_vec = tf.nn.l2_normalize(
            tf.reshape(
                attended_vec,
                shape=[-1, self.heads, self.units * width],
            ),  # shape (N, h, UT')
            axis=-1,
        )
        cosine_similarity = tf.matmul(
            unit_head_vec,
            unit_head_vec,
            transpose_b=True,
        )  # shape (N, h, h)
        return tf.reduce_mean(cosine_similarity)


class MultiHeadSelfAttention(_MultiHeadScaledDotAttention):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
            use_forward_mask: bool = False,
            heads_reg_coeff: float = None,
        ):
        super().__init__(
            units=units,
            output_dim=output_dim,
            heads=heads,
            activation=activation,
            use_bias=use_bias,
            heads_reg_coeff=heads_reg_coeff,
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
        query, key, value = [
            layer(inputs)
            for layer in (self.query_layer, self.key_layer, self.value_layer)
        ]  # shape (N, T, hU)

        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)
            query *= mask[:, :, tf.newaxis]

        attended_vec = self._multihead_attention(
            query=query,
            key=key,
            value=value,
            value_mask=mask,
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

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)


class MultiHeadAttention(_MultiHeadScaledDotAttention):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
            heads_reg_coeff: float = None,
        ):
        super().__init__(
            units=units,
            output_dim=output_dim,
            heads=heads,
            activation=activation,
            use_bias=use_bias,
            heads_reg_coeff=heads_reg_coeff,
        )
        self._input_spec = [tf.keras.layers.InputSpec(ndim=3) for _ in range(2)]

    def build(self, input_shape_tuple):
        if len(input_shape_tuple) != 2:
            raise TypeError("both 'inputs' should be length 2 tuple!")

        # Reference: https://tunz.kr/post/4
        # In order to use glorot uniform with fan_out = units instead of units * heads

        input_shape, encoder_output_shape = input_shape_tuple
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
                [input_shape, encoder_output_shape, encoder_output_shape],
            )
        ]
        super().build(input_shape_tuple)

    def call(
            self,
            inputs_tuple: List[tf.Tensor],
            mask: List[tf.Tensor] = None,
        ) -> tf.Tensor:
        if mask is None:
            mask = [None, None]
        if not (len(inputs_tuple) == len(mask) == 2):
            raise TypeError("both 'inputs' and 'mask' should be length 2 tuple!")

        q, kv = inputs_tuple
        inputs_mask, kv_mask = mask

        query = self.query_layer(q)  # shape (N, T', hU)
        key, value = [
            layer(kv)
            for layer in (self.key_layer, self.value_layer)
        ]  # shape (N, T, hU)

        if kv_mask is not None:
            kv_mask = tf.cast(kv_mask, kv.dtype)  # shape (N, T)

        if inputs_mask is not None:
            query *= tf.cast(inputs_mask, query.dtype)[:, :, tf.newaxis]  # shape (N, T', d_out)

        attended_vec = self._multihead_attention(
            query=query,
            key=key,
            value=value,
            value_mask=kv_mask,
        )
        outputs = self.output_layer(attended_vec)
        return outputs

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask[0]

    def compute_output_shape(self, input_shape_tuple):
        if not (len(input_shape_tuple) == 2):
            raise TypeError("'input_shape_tuple' should be length 2 tuple!")

        input_shape, _ = input_shape_tuple
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)
