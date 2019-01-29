import pytest

import numpy as np
import tensorflow as tf

from talos.networks import Sequential
from ..attention import ScaledDotSelfAttention


@pytest.fixture(scope='module')
def inputs():
    return tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])


@pytest.fixture(scope='module')
def mask():
    return tf.placeholder(dtype=tf.bool, shape=[None, 4])


@pytest.fixture
def layer():
    return ScaledDotSelfAttention(units=6, heads=2)


def test_output_shape(inputs, layer):
    outputs = layer(inputs)
    assert outputs.shape.as_list() == [None, 4, 6 * 2]
    assert layer.compute_output_shape(inputs.shape).as_list() == [None, 4, 6 * 2]


@pytest.mark.parametrize('invalid_inputs', [
    tf.zeros(shape=[2, 3]),
    tf.zeros(shape=[2, 3, 1, 1]),
])
def test_raise_invalid_input_rank(invalid_inputs, layer):
    with pytest.raises(ValueError):
        layer(invalid_inputs)


def test_support_masked_inputs(mocker, inputs, layer):
    masked_inputs = tf.keras.layers.Masking()(inputs)

    # since keras use func inspect, directly mock layer.call will cause side effect
    mock_cast = mocker.spy(tf, 'cast')  # would call if mask is passed
    outputs = layer(masked_inputs)
    assert mock_cast.called
    assert outputs._keras_mask is not None


def test_support_masked_inputs_through_sequential(mocker, inputs, layer):
    seq = Sequential([
        tf.keras.layers.Masking(),
        layer,
    ])

    # since keras use func inspect, directly mock layer.call will cause side effect
    mock_cast = mocker.spy(tf, 'cast')  # would call if mask is passed
    outputs = seq(inputs)
    assert mock_cast.called
    assert outputs._keras_mask is not None


def test_mask_gradients(inputs, mask, layer, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs, mask=mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    sess.run(tf.variables_initializer(var_list=layer.variables))
    grad_batch = sess.run(
        grads,
        feed_dict={
            inputs: [np.random.rand(maxlen, channel) for _ in range(5)],
            mask: mask_batch,
        },
    )
    for mask_sample, grad_sample in zip(mask_batch, grad_batch):
        dropped_section = grad_sample[np.logical_not(mask_sample)]
        assert (dropped_section == 0.).all()
