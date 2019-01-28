import pytest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from ..pooling import GlobalAttentionPooling1D
from ..pooling import GlobalAveragePooling1D


class TestGlobalPooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(tf.float32, [None, 5, 1])

    @pytest.fixture(scope='class')
    def mask(self, inputs):
        maxlen = inputs.shape[1].value
        return tf.placeholder(tf.bool, [None, maxlen])

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAveragePooling1D()

    def test_mask_value(self, sess, inputs, mask, layer):
        outputs = layer(inputs, mask=mask)

        inputs_val = np.array([
            [0., 1., 2., 3., 4.],
            [2., 3., 4., 5., 6.],
        ]).reshape(-1, 5, 1)

        sess.run(tf.variables_initializer(var_list=layer.variables))
        outputs_val = sess.run(
            outputs,
            feed_dict={
                inputs: inputs_val,
                mask: [
                    [True, True, False, True, False],
                    [False, False, False, False, False],
                ],
            },
        )

        expected_outputs_val = np.array([(0. + 1. + 3.) / 3., 0.]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(outputs_val, expected_outputs_val)

    def test_support_masked_inputs(self, inputs, layer):
        masked_inputs = tf.keras.layers.Masking()(inputs)
        with patch.object(layer, 'call') as mock_layer_call:
            layer(masked_inputs)

        kwargs = mock_layer_call.call_args[1]
        assert isinstance(kwargs['mask'], tf.Tensor)  # has passed to layer


class TestGlobalAttentionPooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])

    @pytest.fixture(scope='class')
    def mask(self, inputs):
        maxlen = inputs.shape[1].value
        return tf.placeholder(tf.bool, [None, maxlen])

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAttentionPooling1D(units=3, heads=5)

    def test_output_shape(self, inputs, layer):
        width, channel = inputs.shape.as_list()[1:]
        heads = layer.heads
        outputs = layer(inputs)

        assert outputs.shape.as_list() == [None, heads, channel]
        assert layer.compute_output_shape(inputs.shape).as_list() == [None, heads, channel]
        assert len(layer.losses) == 0

    def test_regularization_losses(self):
        channel, units, heads = 3, 4, 5
        att_pool = GlobalAttentionPooling1D(units=units, heads=heads, reg_coeff=1.0)
        many_inputs = [
            tf.zeros([batch_size, width, channel])
            for batch_size, width in zip([1, 2, 3], [4, 5, 6])
        ]
        # try: can call on any rank 3 tensor with same last dim.
        [att_pool(inputs) for inputs in many_inputs]

        losses = att_pool.losses
        assert len(losses) == len(many_inputs)  # any input has its reg loss
        assert all(loss.shape.as_list() == [] for loss in losses)

    @pytest.mark.parametrize('invalid_inputs', [
        tf.zeros(shape=[2, 3]),
        tf.zeros(shape=[2, 3, 1, 1]),
    ])
    def test_raise_invalid_input_rank(self, invalid_inputs, layer):
        with pytest.raises(ValueError):
            layer(invalid_inputs)

    def test_output_value(self, sess, inputs, mask):
        maxlen, channel = inputs.shape.as_list()[1:]
        # zero init to isolate the grad through weights calculation.
        att_pool = GlobalAttentionPooling1D(units=3, heads=4, kernel_initializer='zeros')
        attended_vec = att_pool(inputs, mask=mask)
        grads = tf.gradients(attended_vec, inputs)[0]  # same shape as inputs

        mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        sess.run(tf.variables_initializer(var_list=att_pool.variables))
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
            assert np.logical_and(0. < attended_section, attended_section < att_pool.heads).all()
            assert (dropped_section == 0.).all()

    def test_support_masked_inputs(self, inputs, layer):
        masked_inputs = tf.keras.layers.Masking()(inputs)
        with patch.object(layer, 'call') as mock_layer_call:
            layer(masked_inputs)

        kwargs = mock_layer_call.call_args[1]
        assert isinstance(kwargs['mask'], tf.Tensor)  # has passed to layer
