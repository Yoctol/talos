from itertools import chain
import pytest

import numpy as np
import tensorflow as tf

from talos.layers import (
    Conv1D,
    Conv2D,
    Dense,
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
)
from ..spectral_norm import add_spectral_norm


@pytest.mark.parametrize('layer,inputs', [
    (Dense(10), tf.zeros([3, 4])),
    (Conv1D(filters=10, kernel_size=3), tf.zeros([3, 4, 5])),
    (Conv2D(filters=10, kernel_size=3), tf.zeros([3, 4, 5, 5])),
    (GRUCell(10), tf.zeros([3, 4])),
    (LSTMCell(10), tf.zeros([3, 4])),
    (GRU(10), tf.zeros([3, 4, 5])),
    (LSTM(10), tf.zeros([3, 4, 5])),
    (tf.keras.Sequential([Dense(10), Dense(10), Dense(10)]), tf.zeros([3, 4])),
])
def test_spectral_norm_for_layer(layer, inputs, sess):
    add_spectral_norm(layer)
    if hasattr(layer, 'state_size'):
        state = layer.get_initial_state(inputs)
        if not isinstance(state, list):
            state = [state]
        layer(inputs, state)
    else:
        layer(inputs)

    kernel_list = recursive_get_kernel_attributes(layer)
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


def recursive_get_kernel_attributes(layer):
    if isinstance(layer, tf.keras.Sequential):
        return list(chain.from_iterable([
            recursive_get_kernel_attributes(layer)
            for layer in layer.layers
        ]))

    if isinstance(layer, tf.keras.layers.RNN):
        return recursive_get_kernel_attributes(layer.cell)

    return [
        attr
        for attr_name, attr in layer.__dict__.items()
        if attr_name.endswith('kernel')
    ]
