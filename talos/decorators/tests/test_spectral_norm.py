import pytest

import tensorflow as tf

from ..spectral_norm import add_spectral_norm


@pytest.mark.parametrize('layer,inputs', [
    (tf.keras.layers.Dense(10), tf.zeros([3, 4])),
    (tf.keras.layers.Conv1D(filters=10, kernel_size=3), tf.zeros([3, 4, 5])),
    (tf.keras.layers.Conv2D(filters=10, kernel_size=3), tf.zeros([3, 4, 5, 5])),
])
def test_spectral_norm_single_kernel(layer, inputs):
    add_spectral_norm(layer)
    layer(inputs)

    assert len(layer.updates) == 1
    assert layer.kernel.name.endswith('_sn:0')


@pytest.mark.parametrize('layer', [
    tf.keras.layers.GRUCell(10),
    tf.keras.layers.LSTMCell(10),
])
def test_spectral_norm_rnn(layer):
    inputs = tf.zeros([3, 4])
    state = layer.get_initial_state(inputs)
    if not isinstance(state, list):
        state = [state]
    add_spectral_norm(layer)
    layer(inputs, state)

    kernel_list = [
        attr
        for attr_name, attr in layer.__dict__.items()
        if attr_name.endswith('kernel')
    ]

    assert len(layer.updates) == len(kernel_list)
    assert [kernel.name.endswith('_sn:0') for kernel in kernel_list]
