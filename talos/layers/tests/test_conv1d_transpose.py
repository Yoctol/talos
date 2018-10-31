import pytest

import tensorflow as tf

from ..conv1d_transpose import Conv1DTranspose


def test_conv1d_transpose():
    dconv1d = Conv1DTranspose(filters=10, kernel_size=5)

    width, channel = 10, 5
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = dconv1d(inputs)

    assert outputs.shape.as_list() == [None, width + dconv1d.kernel_size[0] - 1, dconv1d.filters]


def test_conv1d_transpose_invalid_input():
    dconv1d = Conv1DTranspose(filters=10, kernel_size=5)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 10, 5, 1])
    with pytest.raises(ValueError):
        dconv1d(inputs)
