import pytest

import tensorflow as tf

from ..sequential import Sequential


@pytest.fixture
def sequential():
    return Sequential([
        tf.keras.layers.Embedding(20, 10),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.Dense(5),
        tf.keras.layers.MaxPooling1D(),
    ])


def test_build_sublayers_when_first_called(sequential):
    assert all(not layer.built for layer in sequential.layers)
    inputs = tf.zeros([1, 3], dtype=tf.float32)
    sequential(inputs)
    assert all(layer.built for layer in sequential.layers)


def test_context_manager_work_when_first_called(sequential):
    new_graph = tf.Graph()
    assert new_graph is not tf.get_default_graph()
    assert len(sequential.variables) == 0

    with new_graph.as_default(), tf.variable_scope('scope'):
        inputs = tf.zeros([1, 3], dtype=tf.float32)
        sequential(inputs)

    variables = sequential.variables
    assert len(variables) > 0
    assert all(var.graph is new_graph for var in variables)
    assert all(var.name.startswith('scope') for var in variables)


def test_mask_computed_by_layer_propagate():
    sequential_with_masking = Sequential([
        tf.keras.layers.Masking(),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_with_masking.layers)
    outputs = sequential_with_masking(tf.ones([5, 4, 3]))
    assert outputs._keras_mask is not None


def test_mask_given_in_inputs_propagate():
    sequential_supports_mask = Sequential([
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_supports_mask.layers)
    mask = tf.ones([5, 4], dtype=tf.bool)
    outputs = sequential_supports_mask(
        tf.ones([5, 4, 3]),
        mask=mask,
    )
    assert outputs._keras_mask is mask  # since Dense simply pass it.


def test_masked_inputs_propagate():
    masked_inputs = tf.keras.layers.Masking()(tf.ones([5, 4, 3]))
    sequential_supports_mask = Sequential([
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_supports_mask.layers)
    outputs = sequential_supports_mask(masked_inputs)
    assert outputs._keras_mask is masked_inputs._keras_mask
