import tensorflow as tf


def _valid_rnn_cell(cell):
    if not hasattr(cell, 'call'):
        raise ValueError(
            f'`cell` should have a `call` method.',
        )
    if not hasattr(cell, 'state_size'):
        raise ValueError(
            'The RNN cell should have an attribute `state_size` '
            '(tuple of integers, one integer per RNN state).',
        )


class DynamicRecurrent(tf.keras.layers.Wrapper):

    def __init__(self, cell, return_sequences=False):
        _valid_rnn_cell(cell)
        super().__init__(layer=cell)
        self.return_sequences = return_sequences

    def call(self, inputs, seqlen=None, initial_state=None):
        outputs, state = tf.nn.dynamic_rnn(
            cell=self.layer,
            inputs=inputs,
            sequence_length=seqlen,
            initial_state=initial_state,
            dtype=inputs.dtype,
        )  # shape (N, T, D_out), [(N, D_state), ...]
        if self.return_sequences:
            return outputs
        if hasattr(state, 'h'):
            return state.h
        if hasattr(state, '__len__'):
            return state[0]
        return state

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.as_list()
        if self.return_sequences:
            return tf.TensorShape([input_shape[0], input_shape[1], self.layer.units])
        return tf.TensorShape([input_shape[0], self.layer.units])
