import tensorflow as tf


class GRUCell(tf.keras.layers.GRUCell):

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [super().get_initial_state(inputs, batch_size, dtype)]
