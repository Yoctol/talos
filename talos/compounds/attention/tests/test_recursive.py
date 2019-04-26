import pytest

import numpy as np
import tensorflow as tf

from ..recursive import RelativeAttentionCell


@pytest.fixture(scope='module')
def cell():
    return RelativeAttentionCell(units=3, output_dim=10, heads=5)


@pytest.fixture(scope='module')
def state(inputs):
    maxlen, dim = inputs.shape.as_list()[1:]
    return tf.placeholder(dtype=inputs.dtype, shape=[None, maxlen - 1, dim])


@pytest.fixture(scope='module')
def state_mask(state, mask):
    state_maxlen = state.shape[1].value
    return tf.placeholder(dtype=mask.dtype, shape=[None, state_maxlen])


def test_output_shape(cell, inputs, state):
    length = inputs.shape[1].value
    outputs = cell(inputs, state)
    assert outputs.shape.as_list() == [None, length, cell.output_dim]


def test_mask_gradients(inputs, state, mask, state_mask, cell, sess):
    maxlen, channel = inputs.shape.as_list()[1:]
    state_maxlen = state.shape[1].value

    outputs = cell(inputs, state, mask=mask, state_mask=state_mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    state_mask_val = np.random.choice(2, size=[5, state_maxlen]).astype(np.bool)
    mask_val[:, :2] = True  # to make sure at least 2 True
    state_mask_val[:, :2] = True

    sess.run(tf.variables_initializer(var_list=cell.variables))
    grads_val = sess.run(
        grads,
        feed_dict={
            inputs: np.random.rand(5, maxlen, channel),
            state: np.random.rand(5, state_maxlen, channel),
            mask: mask_val,
            state_mask: state_mask_val,
        },
    )
    assert np.equal(
        grads_val != 0.,
        mask_val[:, :, np.newaxis],
    ).all()


def test_forward_mask_gradients(inputs, state, sess):
    layer = RelativeAttentionCell(units=3, output_dim=10, heads=5, use_forward_mask=True)
    maxlen, channel = inputs.shape.as_list()[1:]
    state_maxlen = state.shape[1].value

    outputs = layer(inputs, state=state)
    grads_list = tf.stack([
        tf.gradients(outputs[:, t], inputs)[0]
        for t in range(outputs.shape[1].value)
    ], axis=1)  # every elements have same shape as inputs
    # shape (N, T, T, U)

    sess.run(tf.variables_initializer(var_list=layer.variables))
    grad_list_val = sess.run(
        grads_list,
        feed_dict={
            inputs: np.random.rand(5, maxlen, channel),
            state: np.random.rand(5, state_maxlen, channel),
        },
    )
    assert np.equal(
        grad_list_val != 0.,  # shape (N, T, T, U)
        np.tril(np.ones([maxlen, maxlen], dtype=np.bool))[:, :, np.newaxis],
        # shape (T, T', 1)
    ).all(), grad_list_val != 0
