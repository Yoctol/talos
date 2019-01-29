import pytest

import numpy as np
import tensorflow as tf

from ..transformer_block import TransformerBlock


@pytest.fixture(scope='module')
def inputs():
    return tf.placeholder(tf.float32, [None, 4, 6])


@pytest.fixture(scope='class')
def mask(inputs):
    maxlen = inputs.shape[1].value
    return tf.placeholder(tf.bool, [None, maxlen])


@pytest.fixture(scope='module')
def layer():
    return TransformerBlock(units=3, heads=2)


def test_output_shape(inputs, layer):
    outputs = layer(inputs)
    assert outputs.shape.as_list() == inputs.shape.as_list()


def test_support_masked_inputs(inputs, layer):
    masked_inputs = tf.keras.layers.Masking()(inputs)

    outputs = layer(masked_inputs)
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
