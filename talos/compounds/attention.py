import numpy as np
import tensorflow as tf

from talos.networks import Model


_LARGE_BIAS = 1e4


class GlobalAttentionPooling1D(Model):
    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(
            self,
            units: int,
            heads: int = 1,
            activation: str = 'tanh',
            use_bias: bool = False,
            reg_coeff: float = 0.,
            **kwargs,
        ):
        super().__init__()
        self.units = units
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias

        self.hidden_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
        )
        self.presoftmax_layer = tf.keras.layers.Dense(units=heads)

        if reg_coeff < 0:
            raise ValueError("reg_coeff can't be negative!")
        self.reg_coeff = reg_coeff

        self.supports_masking = True
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property  # override property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        output_dim = input_shape[-1].value
        self.output_layer = tf.keras.layers.Dense(units=output_dim)
        self._input_spec.axes = {2: output_dim}
        if self.reg_coeff > 0 and self.heads > 1:
            self.identity_matrix = tf.eye(self.heads)
        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            mask: tf.Tensor = None,
        ) -> tf.Tensor:
        # shape (N, T, units)
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]  # shape (N, T, 1)
            inputs *= mask

        hidden_outputs = self.hidden_layer(inputs)  # shape (N, T, U)
        logits = self.presoftmax_layer(hidden_outputs)  # shape (N, T, h)
        logits = self._mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, axis=1)  # shape (N, T, h)

        if self.reg_coeff > 0 and self.heads > 1:
            # shape (N, h, h)
            weights_product = tf.matmul(weights, weights, transpose_a=True)
            penalty = self.reg_coeff * tf.reduce_sum(tf.square(
                weights_product - self.identity_matrix,
            ), axis=[1, 2])  # shape (N)
            batch_penalty = tf.reduce_mean(penalty)
            # NOTE directly call self.add_loss won't work...
            # keras Model do some complicated check...
            self.output_layer.add_loss(batch_penalty)

        # shape (N, h, input_dim)
        attended_vec = tf.matmul(weights, inputs, transpose_a=True)
        inputs_dim = inputs.shape[-1].value
        flatten_attended_vec = tf.reshape(attended_vec, [-1, self.heads * inputs_dim])
        outputs = self.output_layer(flatten_attended_vec)
        return outputs

    def _mask_logits(self, logits, mask):
        if mask is not None:
            bias = (1. - mask) * _LARGE_BIAS  # shape (N, T, 1)
            logits -= bias

        return logits

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], input_shape[2]])

    def compute_mask(self, inputs, mask):
        return None


class ScaledDotSelfAttention(Model):

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            activation: str = None,
            use_bias: bool = False,
            use_forward_mask: bool = False,
        ):
        super().__init__()
        self.units = units
        self.output_dim = output_dim
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias
        self.use_forward_mask = use_forward_mask

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
        query, key, value = [
            layer(inputs)
            for layer in (self.query_layer, self.key_layer, self.value_layer)
        ]  # shape (N, T, hU)
        if self.heads > 1:
            # shape (N, T, hU) -> (N, hU, T) -> (N, h, U, T)
            query, key, value = [
                tf.reshape(
                    tf.transpose(tensor, perm=[0, 2, 1]),
                    shape=[-1, self.heads, self.units, width],
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
                attended_vec, shape=[-1, self.heads * self.units, width])  # shape (N, hU, T')
            attended_vec = tf.transpose(attended_vec, perm=[0, 2, 1])  # shape (N, T', hU)
        else:
            # (N, T', T) * (N, T, U) -> (N, T', U)
            attended_vec = tf.matmul(weights, value)

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

        if mask is not None:
            bias = (1. - mask) * _LARGE_BIAS
            if self.heads > 1:
                logits -= bias[:, tf.newaxis, tf.newaxis, :]  # shape (N, 1, 1, T)
            else:
                logits -= bias[:, tf.newaxis, :]  # shape (N, 1, T)

        return logits

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
