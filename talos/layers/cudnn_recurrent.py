import tensorflow as tf


class CuDNNGRU(tf.keras.layers.CuDNNGRU):

    # remove the mask arg to avoid mask fed by Sequential
    def call(self, inputs, training=None, initial_state=None):
        return super.call(inputs, training=training, initial_state=initial_state)


class CuDNNLSTM(tf.keras.layers.CuDNNLSTM):

    # remove the mask arg to avoid mask fed by Sequential
    def call(self, inputs, training=None, initial_state=None):
        return super.call(inputs, training=training, initial_state=initial_state)
