import pytest

import numpy as np
import tensorflow as tf

from ..conv1d_transpose import Conv1DTranspose


def test_output_shape_valid_padding():
    width, channel = 10, 4
    filters, kernel_size = 3, 5
    dconv1d = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='valid')
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = dconv1d(inputs)

    assert outputs.shape.as_list() == [None, width + kernel_size - 1, filters]
    config = dconv1d.get_config()
    assert config['filters'] == filters
    assert config['kernel_size'] == (kernel_size, )


def test_output_shape_same_padding():
    width, channel = 10, 4
    filters, kernel_size = 3, 5
    dconv1d = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='same')
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = dconv1d(inputs)

    assert outputs.shape.as_list() == [None, width, filters]
    config = dconv1d.get_config()
    assert config['filters'] == filters
    assert config['kernel_size'] == (kernel_size, )


def test_output_value_valid_padding():
    width, channel = 3, 1
    dconv1d = Conv1DTranspose(
        filters=1,
        kernel_size=3,
        kernel_initializer='ones',
        padding='valid',
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = dconv1d(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_val = sess.run(
            outputs,
            feed_dict={inputs: np.array([[[1.], [2.], [3.]]])},
        )

    expected_outputs_val = np.sum([
        np.array([[[1.], [1.], [1.], [0.], [0.]]]),
        np.array([[[0.], [2.], [2.], [2.], [0.]]]),
        np.array([[[0.], [0.], [3.], [3.], [3.]]]),
    ], axis=0)
    np.testing.assert_array_almost_equal(
        outputs_val,
        expected_outputs_val,
    )


def test_output_value_same_padding():
    width, channel = 3, 1
    dconv1d = Conv1DTranspose(
        filters=1,
        kernel_size=3,
        kernel_initializer='ones',
        padding='same',
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = dconv1d(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_val = sess.run(
            outputs,
            feed_dict={inputs: np.array([[[1.], [2.], [3.]]])},
        )

    expected_outputs_val = np.sum([
        np.array([[[1.], [1.], [0.]]]),
        np.array([[[2.], [2.], [2.]]]),
        np.array([[[0.], [3.], [3.]]]),
    ], axis=0)
    np.testing.assert_array_almost_equal(
        outputs_val,
        expected_outputs_val,
    )


def test_invalid_input_rank():
    rank4_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 10, 5, 1])
    with pytest.raises(ValueError):
        Conv1DTranspose(filters=10, kernel_size=5)(rank4_inputs)


def test_invalid_input_shape():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 10, 5])
    different_shape_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 10, 6])
    dconv1d = Conv1DTranspose(filters=10, kernel_size=5)
    dconv1d(inputs)
    with pytest.raises(ValueError):
        dconv1d(different_shape_inputs)
