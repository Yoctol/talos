import pytest

import tensorflow as tf

from ..layers import Conv1D, Dense


@pytest.fixture(scope='module')
def dense_layer():
    return Dense(units=10)


@pytest.fixture(scope='module')
def conv1d_layer():
    return Conv1D(filters=10, kernel_size=3)


def test_dense(dense_layer):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 20])
    outputs = dense_layer(inputs)
    assert outputs.shape.as_list() == [None, 10]


def test_dense_rank3(dense_layer):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5, 20])
    outputs = dense_layer(inputs)
    assert outputs.shape.as_list() == [None, 5, 10]


def test_dense_invalid_input_dim(dense_layer):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    with pytest.raises(ValueError):
        dense_layer(inputs)


def test_invalid_id_arguments():
    with pytest.raises(KeyError):
        Dense(10, activation="nonexistent")
    with pytest.raises(KeyError):
        Dense(10, kernel_initializer="nonexistent")
    with pytest.raises(KeyError):
        Dense(10, bias_initializer="nonexistent")


def test_conv1d(conv1d_layer):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5, 20])
    outputs = conv1d_layer(inputs)
    assert outputs.shape.as_list() == [None, 3, 10]
