import pytest

import numpy as np
import tensorflow as tf

from ..recursive import RelativeAttentionCell


@pytest.fixture(scope='module')
def cell():
    return RelativeAttentionCell(units=3, output_dim=10, heads=5)


@pytest.fixture(scope='module')
def state(inputs):
    return tf.placeholder(dtype=inputs.dtype, shape=inputs.shape.as_list())


@pytest.fixture(scope='module')
def state_mask(mask):
    return tf.placeholder(dtype=mask.dtype, shape=mask.shape.as_list())


def test_output_shape(cell, inputs, state):
    length = inputs.shape[1].value
    outputs = cell(inputs, state)
    assert outputs.shape.as_list() == [None, length, cell.output_dim]


def test_mask_gradients(inputs, state, mask, state_mask, cell, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = cell(inputs, state, mask=mask, state_mask=state_mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    state_mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    mask_val[:, :2] = True  # to make sure at least 2 True
    state_mask_val[:, :2] = True

    sess.run(tf.variables_initializer(var_list=cell.variables))
    grads_val = sess.run(
        grads,
        feed_dict={
            inputs: np.random.rand(5, maxlen, channel),
            state: np.random.rand(5, maxlen, channel),
            mask: mask_val,
            state_mask: state_mask_val
        },
    )
    assert np.equal(
        grads_val != 0.,
        mask_val[:, :, np.newaxis],
    ).all()
