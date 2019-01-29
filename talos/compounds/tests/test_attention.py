import pytest

import tensorflow as tf

from talos.networks import Sequential
from ..attention import ScaledDotSelfAttention


@pytest.fixture(scope='module')
def inputs():
    return tf.zeros([5, 4, 3])


@pytest.fixture
def layer():
    return ScaledDotSelfAttention(units=6, heads=2)


def test_output_shape(inputs, layer):
    outputs = layer(inputs)
    assert outputs.shape.as_list() == [5, 4, 6 * 2]
    assert layer.compute_output_shape(inputs.shape) == [5, 4, 6 * 2]


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
