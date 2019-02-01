import pytest

import numpy as np
import tensorflow as tf

from ..layer_norm import LayerNormalization


@pytest.fixture
def layer():
    return LayerNormalization()


@pytest.fixture(scope='module', params=[
    tf.zeros([3, 5]),
    tf.zeros([3, 4, 5]),
])
def inputs(request):
    return request.param


def test_output_shape(layer, inputs):
    outputs = layer(inputs)
    assert outputs.shape.as_list() == inputs.shape.as_list()
    assert layer.compute_output_shape(inputs.shape) == inputs.shape.as_list()


def test_no_variable_input_spec_without_scale_and_center(inputs):
    layer = LayerNormalization(scale=False, center=False)
    layer(inputs)
    assert len(layer.variables) == 0
    assert layer.input_spec is None


def test_variables_input_spec_with_scale_and_center(layer, inputs, sess):
    layer(inputs)
    assert len(layer.variables) == len(layer.trainable_variables) == 2

    input_dim = inputs.shape[-1].value
    assert layer.beta.shape.as_list() == [input_dim]
    assert layer.gamma.shape.as_list() == [input_dim]

    sess.run(tf.variables_initializer(var_list=layer.variables))
    assert (sess.run(layer.beta) == 0.).all()  # default initializer
    assert (sess.run(layer.gamma) == 1.).all()  # default initializer
    assert layer.input_spec.axes == {-1: input_dim}


@pytest.mark.parametrize('input_val', [
    np.array([[1, 2, 3], [4, 5, 6]]),
])
def test_output_value(input_val, sess):
    layer = LayerNormalization(
        beta_initializer=tf.random_normal_initializer(),
        gamma_initializer=tf.random_normal_initializer(),
    )
    inputs = tf.constant(input_val, dtype=tf.float32)
    outputs = layer(inputs)

    sess.run(tf.variables_initializer(var_list=layer.variables))

    output_val, beta, gamma = sess.run([outputs, layer.beta, layer.gamma])
    mean = np.mean(input_val, axis=1, keepdims=True)
    std = np.std(input_val, axis=1, keepdims=True)
    expected_output_val = gamma * (input_val - mean) / std + beta
    np.testing.assert_array_almost_equal(
        output_val,
        expected_output_val,
    )
