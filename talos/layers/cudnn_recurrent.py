import tensorflow as tf


class CuDNNGRU(tf.keras.layers.CuDNNGRU):

    # remove the mask arg to avoid mask fed by Sequential
    def call(self, inputs, training=None, initial_state=None):
        return super().call(inputs, training=training, initial_state=initial_state)

    @property
    def trainable_weights(self):
        return super(tf.keras.layers.RNN, self).trainable_weights

    @property
    def non_trainable_weights(self):
        return super(tf.keras.layers.RNN, self).non_trainable_weights


class CuDNNLSTM(tf.keras.layers.CuDNNLSTM):

    # remove the mask arg to avoid mask fed by Sequential
    def call(self, inputs, training=None, initial_state=None):
        return super().call(inputs, training=training, initial_state=initial_state)

    @property
    def trainable_weights(self):
        return super(tf.keras.layers.RNN, self).trainable_weights

    @property
    def non_trainable_weights(self):
        return super(tf.keras.layers.RNN, self).non_trainable_weights
