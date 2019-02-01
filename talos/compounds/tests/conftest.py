import pytest

import tensorflow as tf


@pytest.fixture(scope='session')
def inputs():
    return tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])


@pytest.fixture(scope='session')
def mask():
    return tf.placeholder(dtype=tf.bool, shape=[None, 4])


@pytest.fixture(scope='session')
def masked_inputs(inputs):
    outputs = tf.keras.layers.Masking()(inputs)
    # don't change this part!!!!
    assert isinstance(outputs._keras_mask, tf.Tensor)
    return outputs
