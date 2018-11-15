import tensorflow as tf


class GlobalAttentionPooling1D(tf.keras.Model):
    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(
            self,
            units: int,
            heads: int = 1,
            activation: str = 'tanh',
            use_bias: bool = False,
            reg_coeff: float = 0.,
        ):
        super().__init__()
        if reg_coeff < 0:
            raise ValueError("reg_coeff can't be negative!")
        self.reg_coeff = reg_coeff
        self._identity_matrix = None
        self.units = units
        self.heads = heads

        self.candidate_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            name='candidate_layer',
        )
        self.softmax_layer = tf.keras.layers.Dense(
            units=heads,
            activation=lambda logits: tf.nn.softmax(logits, axis=1),
            use_bias=False,
            name='softmax_layer',
        )

    def call(
            self,
            inputs: tf.Tensor,
            seqlen: tf.Tensor = None,
        ) -> tf.Tensor:
        if inputs.shape.ndims != 3:
            raise ValueError("Input of GlobalAttentionPooling1D should be rank 3!")
        # shape (N, T, units)
        hidden_outputs = self.candidate_layer(inputs)
        # shape (N, T, Head)
        weights = self.softmax_layer(hidden_outputs)
        if seqlen is not None:
            # Renormalize for lower seqlen
            maxlen = inputs.shape[1].value
            # shape (N, T)
            mask = tf.sequence_mask(seqlen, maxlen=maxlen, dtype=tf.float32)
            weights *= tf.expand_dims(mask, axis=2)
            weights /= tf.reduce_sum(weights, axis=1, keepdims=True)

        if self.reg_coeff > 0:
            weights_product = tf.matmul(weights, weights, transpose_a=True)
            identity_matrix = self._get_identity_matrix()
            penalty = self.reg_coeff * tf.reduce_sum(tf.square(
                weights_product - identity_matrix,
            ))
            self.softmax_layer.add_loss(penalty)
        # shape (N, Head, input_dim)
        outputs = tf.matmul(weights, inputs, transpose_a=True)
        return outputs

    def _get_identity_matrix(self):
        if self._identity_matrix is None:
            self._identity_matrix = tf.eye(self.heads, batch_shape=[1])
        return self._identity_matrix
