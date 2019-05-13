from itertools import chain
import pytest

import numpy as np
import tensorflow as tf

from talos.compounds import TransformerBlock
from talos.layers import (
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNN,
    StackedRNNCells,
)
from talos.networks import Sequential
from ..spectral_norm import add_spectral_norm


def test_grad_value(sess):
    x = tf.random_normal([3, 5])
    layer = Dense(10, use_bias=False)
    add_spectral_norm(layer)
    y = layer(x)

    W = layer.trainable_variables[0]
    W_sn = layer.kernel
    dW, dW_sn = tf.gradients(y, [W, W_sn])

    graph = sess.graph
    sn = graph.get_tensor_by_name(f'{W.op.name}/singular_value:0')
    # use the value after updated
    new_u = graph.get_tensor_by_name(f'{W.op.name}/new_left_singular_vector:0')
    new_v = graph.get_tensor_by_name(f'{W.op.name}/new_right_singular_vector:0')

    sess.run(tf.variables_initializer(layer.variables))
    dW_val, dW_sn_val, W_sn_val, sn_val, u, v = sess.run([
        dW, dW_sn, W_sn, sn, new_u, new_v])

    eps = tf.keras.backend.epsilon()
    # Formula Reference: https://hackmd.io/HHWrnmbWSXKb_DIFmHKtAA
    expected_dW = (dW_sn_val - u * v[:, 0] * np.sum(W_sn_val * dW_sn_val)) / (sn_val + eps)
    np.testing.assert_array_almost_equal(dW_val, expected_dW, decimal=4)


@pytest.mark.parametrize('layer, inputs', [
    (Dense(10), tf.zeros([3, 4])),
    (Conv1D(filters=10, kernel_size=3), tf.zeros([3, 4, 5])),
    (Conv2D(filters=10, kernel_size=3), tf.zeros([3, 4, 5, 5])),
    (GRUCell(10), tf.zeros([3, 4])),
    (LSTMCell(10), tf.zeros([3, 4])),
    (GRU(10), tf.zeros([3, 4, 5])),
    (LSTM(10), tf.zeros([3, 4, 5])),
    (Sequential([
        Sequential([Dense(10), Dense(10)]),
        LSTM(10),
    ]), tf.zeros([3, 4, 5])),
    (RNN(StackedRNNCells([
        LSTMCell(5),
        LSTMCell(5),
    ])), tf.zeros([3, 4, 5])),
    (Bidirectional(LSTM(10)), tf.zeros([3, 4, 5])),
    (TransformerBlock(5, heads=4), tf.zeros([3, 4, 5])),
])
def test_add_spectral_norm(layer, inputs, sess):
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
    # Since norm come from division
    assert all([kernel.op.type == 'RealDiv' for kernel in kernel_list])

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
    if isinstance(layer, tf.keras.Model):
        return list(chain.from_iterable([
            recursive_get_kernel_attributes(layer)
            for layer in layer.layers
        ]))

    if isinstance(layer, tf.keras.layers.StackedRNNCells):
        return list(chain.from_iterable([
            recursive_get_kernel_attributes(cell)
            for cell in layer.cells
        ]))

    if isinstance(layer, tf.keras.layers.Bidirectional):
        return list(chain.from_iterable([
            recursive_get_kernel_attributes(layer.forward_layer),
            recursive_get_kernel_attributes(layer.backward_layer),
        ]))

    if isinstance(layer, tf.keras.layers.Wrapper):
        return recursive_get_kernel_attributes(layer.layer)

    if isinstance(layer, tf.keras.layers.RNN):
        return recursive_get_kernel_attributes(layer.cell)

    return [
        attr
        for attr_name, attr in layer.__dict__.items()
        if attr_name.endswith('kernel')
    ]
