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
            **kwargs,
        ):
        self.candidate_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=False,
            name='candidate_layer',
        )
        self.presoftmax_layer = tf.keras.layers.Dense(
            units=heads,
            activation=None,
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
            raise ValueError()
        # shape (N, T, units)
        hidden_outputs = self.candidate_layer(inputs)
        # shape (N, T, Head)
        logits = self.presoftmax_layer(hidden_outputs)
        weights = tf.nn.softmax(logits, axis=1)
        if seqlen is not None:
            mask = tf.to_float(tf.sequence_mask(seqlen))
            weights *= tf.expand_dims(mask, axis=-1)
            weights /= tf.reduce_sum(weights)

        # shape (N, Head, input_dim)
        outputs = tf.matmul(weights, inputs, transpose_a=True)
        return outputs
