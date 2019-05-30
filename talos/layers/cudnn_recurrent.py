import tensorflow as tf


class CuDNNGRU(tf.keras.layers.CuDNNGRU):

    def __init__(
            self,
            units,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            **kwargs,
        ):
        super().__init__(
            units,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )
        self.supports_masking = return_sequences

    # bypass the mask
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(inputs, training=training, initial_state=initial_state)

    @property
    def trainable_weights(self):
        return super(tf.keras.layers.RNN, self).trainable_weights

    @property
    def non_trainable_weights(self):
        return super(tf.keras.layers.RNN, self).non_trainable_weights


class CuDNNLSTM(tf.keras.layers.CuDNNLSTM):

    def __init__(
            self,
            units,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            **kwargs,
        ):
        super().__init__(
            units,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )
        # if return_sequences, bypass the mask
        self.supports_masking = return_sequences

    # bypass the mask
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(inputs, training=training, initial_state=initial_state)

    @property
    def trainable_weights(self):
        return super(tf.keras.layers.RNN, self).trainable_weights

    @property
    def non_trainable_weights(self):
        return super(tf.keras.layers.RNN, self).non_trainable_weights
