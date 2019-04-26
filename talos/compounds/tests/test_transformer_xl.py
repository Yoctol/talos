import pytest

import numpy as np
import tensorflow as tf

from ..transformer_xl import TransformerXL


@pytest.fixture(scope='module', params=[False, True])
def layer(request):
    bidirectional = request.param
    return TransformerXL(block_size=2, units=3, heads=5, bidirectional=bidirectional)


def test_output_shape(layer, inputs):
    outputs = layer(inputs)
    assert outputs.shape.as_list() == inputs.shape.as_list()


def test_mask_gradients(inputs, mask, layer, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs, mask=mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    mask_val[:, :2] = True  # to make sure at least 2 True

    sess.run(tf.variables_initializer(var_list=layer.variables))
    grads_val = sess.run(
        grads,
        feed_dict={
            inputs: np.random.rand(5, maxlen, channel),
            mask: mask_val,
        },
    )
    assert np.equal(
        grads_val != 0.,
        mask_val[:, :, np.newaxis],
    ).all()


@pytest.mark.parametrize('layer', [
    TransformerXL(block_size=2, units=3, heads=2, use_forward_mask=True),
    TransformerXL(block_size=2, units=3, heads=2, use_forward_mask=True, state_gradient=True),
    TransformerXL(block_size=2, units=3, heads=2, bidirectional=True),
])
def test_blocklevel_gradients(layer, sess):
    inputs = tf.random_normal([5, 5, 4])
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs)
    grads_list = tf.stack([
        tf.gradients(outputs[:, t], inputs)[0]
        for t in range(maxlen)
    ], axis=1)  # every elements have same shape as inputs
    # shape (N, T, T, U)

    sess.run(tf.variables_initializer(var_list=layer.variables))
    grad_list_val = sess.run(grads_list)
    attention_map = generate_attention_map(
        maxlen,
        layer.block_size,
        bidirectional=layer.bidirectional,
        forward_mask=layer.use_forward_mask,
        state_gradient=layer.state_gradient,
    )
    assert np.equal(
        grad_list_val != 0.,  # shape (N, T, T, U)
        attention_map[:, :, np.newaxis],  # shape (T, T, 1)
    ).all(), np.any(grad_list_val, axis=-1)[0]


def generate_attention_map(maxlen, block_size, bidirectional, forward_mask, state_gradient):
    att_map = np.zeros([maxlen, maxlen], dtype=np.bool)
    for t in range(0, maxlen, block_size):
        # diagonal block
        if forward_mask:
            att_map[t: t + block_size, t: t + block_size] = np.tril(
                np.ones(
                    [min(block_size, maxlen - t), min(block_size, maxlen - t)],
                    dtype=np.bool,
                ),
            )
        else:
            att_map[t: t + block_size, t: t + block_size] = True

        if state_gradient and t >= block_size:
            att_map[t: t + block_size, t - block_size: t] = True  # section from previous block

    if bidirectional:
        att_map = np.logical_or(
            att_map,
            att_map[::-1, ::-1],
        )
    return att_map
