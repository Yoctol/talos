import tensorflow as tf

from .module import Module


class GlobalAttentionPooling1D(Module):
    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(
            self,
            units,
            heads=1,
            activation='tanh',
            use_bias=False,
            **kwargs,
        ):
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
            name='presoftmax_layer',
        )
        super().__init__(sub_layers=[self.candidate_layer, self.presoftmax_layer], **kwargs)

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

        # shape (N, Head, input_dim)
        outputs = tf.matmul(weights, inputs, transpose_a=True)
        return outputs
