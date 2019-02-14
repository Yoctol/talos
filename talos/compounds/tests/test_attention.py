import pytest

import numpy as np
import tensorflow as tf

from ..attention import (
    GlobalAttentionPooling1D,
    ScaledDotSelfAttention,
)


@pytest.fixture(params=[
    ScaledDotSelfAttention(units=3, heads=2, output_dim=5),
    ScaledDotSelfAttention(units=3, heads=1, output_dim=5),
])
def layer(request):
    return request.param


def test_output_shape(inputs, layer):
    batch_size, maxlen = inputs.shape.as_list()[:2]
    output_dim = layer.output_dim
    outputs = layer(inputs)
    expected_shape = [batch_size, maxlen, output_dim]
    assert outputs.shape.as_list() == expected_shape
    assert layer.compute_output_shape(inputs.shape).as_list() == expected_shape


@pytest.mark.parametrize('invalid_inputs', [
    tf.zeros(shape=[2, 3]),
    tf.zeros(shape=[2, 3, 1, 1]),
])
def test_raise_invalid_input_rank(invalid_inputs, layer):
    with pytest.raises(ValueError):
        layer(invalid_inputs)


def test_masked_inputs_propagate(mocker, masked_inputs, layer):
    # since keras use func inspect, directly mock layer.call will cause side effect
    assert isinstance(masked_inputs._keras_mask, tf.Tensor)
    mock_cast = mocker.spy(tf, 'cast')  # would call if mask is passed
    outputs = layer(masked_inputs)
    assert mock_cast.called
    assert outputs._keras_mask is masked_inputs._keras_mask


def test_mask_gradients(inputs, mask, layer, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs, mask=mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    sess.run(tf.variables_initializer(var_list=layer.variables))
    grads_batch = sess.run(
        grads,
        feed_dict={
            inputs: [np.random.rand(maxlen, channel) for _ in range(5)],
            mask: mask_batch,
        },
    )
    for mask_sample, grad_sample in zip(mask_batch, grads_batch):
        attended_section = grad_sample[mask_sample]
        dropped_section = grad_sample[np.logical_not(mask_sample)]
        assert (attended_section != 0.).all()
        assert (dropped_section == 0.).all()


@pytest.mark.parametrize('layer', [
    ScaledDotSelfAttention(units=3, heads=2, output_dim=5, use_forward_mask=True),
    ScaledDotSelfAttention(units=3, heads=1, output_dim=5, use_forward_mask=True),
])
def test_forward_mask_gradients(inputs, layer, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs)
    grads_list = [
        tf.gradients(
            outputs[:, t],
            inputs,
        )[0]
        for t in range(maxlen)
    ]  # every elements have same shape as inputs

    sess.run(tf.variables_initializer(var_list=layer.variables))
    grad_list_val = sess.run(
        grads_list,
        feed_dict={inputs: np.random.rand(1, maxlen, channel)},
    )
    for t, grad_of_output_t in enumerate(grad_list_val):
        attended_section = grad_of_output_t[:, :t + 1]
        dropped_section = grad_of_output_t[:, t + 1:]
        assert (attended_section != 0.).all()
        assert (dropped_section == 0.).all()


class TestGlobalAttentionPooling1D:

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAttentionPooling1D(units=3, heads=5)

    def test_output_shape(self, inputs, layer):
        width, channel = inputs.shape.as_list()[1:]
        outputs = layer(inputs)

        assert outputs.shape.as_list() == [None, channel]
        assert layer.compute_output_shape(inputs.shape).as_list() == [None, channel]
        assert len(layer.losses) == 0

    def test_regularization_losses(self):
        channel, units, heads = 3, 4, 5
        layer = GlobalAttentionPooling1D(units=units, heads=heads, reg_coeff=1.0)
        many_inputs = [
            tf.zeros([batch_size, width, channel])
            for batch_size, width in zip([1, 2, 3], [4, 5, 6])
        ]
        # try: can call on any rank 3 tensor with same last dim.
        outputs = [layer(inputs) for inputs in many_inputs]

        losses = layer.losses
        assert len(losses) == len(outputs)  # any input has its reg loss
        assert all(loss.shape.as_list() == [] for loss in losses)

    @pytest.mark.parametrize('invalid_inputs', [
        tf.zeros(shape=[2, 3]),
        tf.zeros(shape=[2, 3, 1, 1]),
    ])
    def test_raise_invalid_input_rank(self, invalid_inputs, layer):
        with pytest.raises(ValueError):
            layer(invalid_inputs)

    def test_masked_inputs_used(self, mocker, masked_inputs, layer):
        # since keras use func inspect, directly mock layer.call will cause side effect
        assert isinstance(masked_inputs._keras_mask, tf.Tensor)
        mock_cast = mocker.spy(tf, 'cast')  # would call if mask is passed
        layer(masked_inputs)
        assert mock_cast.called

    def test_mask_gradients(self, inputs, mask, layer, sess):
        maxlen, channel = inputs.shape.as_list()[1:]
        attended_vec = layer(inputs, mask=mask)
        grads = tf.gradients(attended_vec, inputs)[0]  # same shape as inputs

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
            attended_section = grad_sample[mask_sample]
            dropped_section = grad_sample[np.logical_not(mask_sample)]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()
