import pytest

import tensorflow as tf

from ..layers import Dense


@pytest.fixture(scope='module')
def dense_layer():
    return Dense(units=10)


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
