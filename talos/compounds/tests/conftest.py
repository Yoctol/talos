import pytest

import tensorflow as tf


@pytest.fixture(scope='session')
def inputs():
    return tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])


@pytest.fixture(scope='session')
def mask():
    return tf.placeholder(dtype=tf.bool, shape=[None, 4])
