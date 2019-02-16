import pytest

import numpy as np
import tensorflow as tf

from ..spectral_norm import (
    add_spectral_norm,
    add_spectral_norm_for_layer,
)


@pytest.mark.parametrize('layer,inputs', [
    (tf.keras.layers.Dense(10), tf.zeros([3, 4])),
    (tf.keras.layers.Conv1D(filters=10, kernel_size=3), tf.zeros([3, 4, 5])),
    (tf.keras.layers.Conv2D(filters=10, kernel_size=3), tf.zeros([3, 4, 5, 5])),
    (tf.keras.layers.GRUCell(10), tf.zeros([3, 4])),
    (tf.keras.layers.LSTMCell(10), tf.zeros([3, 4])),
])
def test_spectral_norm_for_layer(layer, inputs, sess):
    add_spectral_norm_for_layer(layer)
    if hasattr(layer, 'state_size'):
        state = layer.get_initial_state(inputs)
        if not isinstance(state, list):
            state = [state]
        layer(inputs, state)
    else:
        layer(inputs)

    kernel_list = [
        attr
        for attr_name, attr in layer.__dict__.items()
        if attr_name.endswith('kernel')
    ]
    u_vector_list = layer.non_trainable_variables

    assert len(layer.updates) == len(u_vector_list) == len(kernel_list)
    assert all([kernel.name.endswith('_sn:0') for kernel in kernel_list])

    sess.run(tf.variables_initializer(layer.variables))

    u_vector_val_list = sess.run(u_vector_list)
    sess.run(layer.updates)
    updated_u_vector_val_list = sess.run(u_vector_list)

    for u_vector_val, updated_u_vector_val in zip(u_vector_val_list, updated_u_vector_val_list):
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                u_vector_val,
                updated_u_vector_val,
            )


@pytest.mark.parametrize('rnn_layer', [
    tf.keras.layers.GRU(10),
    tf.keras.layers.LSTM(10),
])
def test_add_spectral_norm(rnn_layer):
    add_spectral_norm(rnn_layer)
    inputs = tf.zeros([3, 4, 5])
    rnn_layer(inputs)

    kernel_list = [
        attr
        for attr_name, attr in rnn_layer.cell.__dict__.items()
        if attr_name.endswith('kernel')
    ]
    u_vector_list = rnn_layer.non_trainable_variables

    assert len(rnn_layer.updates) == len(u_vector_list) == len(kernel_list)
    assert all([kernel.name.endswith('_sn:0') for kernel in kernel_list])
