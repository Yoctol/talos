import abc
import pytest

import numpy as np
import tensorflow as tf

from .. import (
    GlobalAttentionPooling1D,
    MultiHeadSelfAttention,
    MultiHeadAttention,
)


class AttentionTestTemplate(abc.ABC):

    def test_output_shape(self, inputs, layer):
        outputs = layer(inputs)
        expected_shape = self.get_expected_shape(layer, inputs)
        assert outputs.shape.as_list() == expected_shape
        assert layer.compute_output_shape(inputs.shape).as_list() == expected_shape

    @pytest.mark.parametrize('invalid_inputs', [
        tf.zeros(shape=[2, 3]),
        tf.zeros(shape=[2, 3, 1, 1]),
    ])
    def test_raise_invalid_input_rank(self, invalid_inputs, layer):
        with pytest.raises(ValueError):
            layer(invalid_inputs)

    @abc.abstractmethod
    def get_expected_shape(self, layer, inputs):
        pass

    def test_masked_inputs_used(self, mocker, masked_inputs, layer):
        # since keras use func inspect, directly mock layer.call will cause side effect
        mock_cast = mocker.spy(tf, 'cast')  # would call if mask is passed
        layer(masked_inputs)
        assert mock_cast.called

    def test_mask_gradients(self, inputs, mask, layer, sess):
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


class TestGlobalAttentionPooling1D(AttentionTestTemplate):

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAttentionPooling1D(units=3, heads=5)

    def get_expected_shape(self, layer, inputs):
        return [inputs.shape[0].value, inputs.shape[2].value]

    def test_regularization_losses(self, inputs):
        layer = GlobalAttentionPooling1D(units=3, heads=4, reg_coeff=1.0)

        for _ in range(3):
            layer(inputs)

        losses = layer.losses
        assert len(losses) == 3  # any input has its reg loss
        assert all(loss.shape.ndims == 0 for loss in losses)


class TestMultiHeadSelfAttention(AttentionTestTemplate):

    @pytest.fixture(params=[
        MultiHeadSelfAttention(units=3, heads=2, output_dim=5),
        MultiHeadSelfAttention(units=3, heads=1, output_dim=5),
    ])
    def layer(self, request):
        return request.param

    def get_expected_shape(self, layer, inputs):
        return inputs.shape.as_list()[:2] + [layer.output_dim]

    def test_masked_inputs_propagate(self, mocker, masked_inputs, layer):
        outputs = layer(masked_inputs)
        assert outputs._keras_mask is masked_inputs._keras_mask

    @pytest.mark.parametrize('layer', [
        MultiHeadSelfAttention(units=3, heads=2, output_dim=5, use_forward_mask=True),
        MultiHeadSelfAttention(units=3, heads=1, output_dim=5, use_forward_mask=True),
    ])
    def test_forward_mask_gradients(self, inputs, layer, sess):
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
            feed_dict={inputs: np.random.rand(5, maxlen, channel)},
        )
        for t, grad_of_output_t in enumerate(grad_list_val):
            attended_section = grad_of_output_t[:, :t + 1]
            dropped_section = grad_of_output_t[:, t + 1:]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()


class TestMultiHeadAttention:

    @pytest.fixture(scope='class')
    def encoder_output(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, 5, 10])

    @pytest.fixture(scope='class')
    def encoder_output_mask(self):
        return tf.placeholder(dtype=tf.bool, shape=[None, 5])

    @pytest.fixture(scope='class')
    def layer(self):
        return MultiHeadAttention(units=3, heads=2, output_dim=5)

    def test_multihead_attention(self, inputs, encoder_output, layer):
        outputs_tuple = layer([inputs, encoder_output])
        assert outputs_tuple[0].shape.as_list() == inputs.shape.as_list()[:2] + [layer.output_dim]
        assert outputs_tuple[1] is encoder_output

    def test_mask_gradients(self, inputs, mask, encoder_output, encoder_output_mask, layer, sess):
        maxlen, channel = inputs.shape.as_list()[1:]
        maxlen_encoder, channel_encoder = encoder_output.shape.as_list()[1:]

        outputs, _ = layer(
            [inputs, encoder_output],
            mask=[mask, encoder_output_mask],
        )
        input_grads, encoder_grads = tf.gradients(outputs, [inputs, encoder_output])

        mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        encoder_mask_batch = np.random.choice(2, size=[5, maxlen_encoder]).astype(np.bool)

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grads_batch, encoder_grads_batch = sess.run(
            [input_grads, encoder_grads],
            feed_dict={
                inputs: [np.random.rand(maxlen, channel) for _ in range(5)],
                encoder_output: [np.random.rand(maxlen_encoder, channel_encoder) for _ in range(5)],
                mask: mask_batch,
                encoder_output_mask: encoder_mask_batch,
            },
        )
        for mask_sample, grad_sample, encoder_mask_sample, encoder_grad_sample in zip(
                mask_batch,
                grads_batch,
                encoder_mask_batch,
                encoder_grads_batch,
            ):
            attended_section = grad_sample[mask_sample]
            dropped_section = grad_sample[np.logical_not(mask_sample)]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()

            attended_section = encoder_grad_sample[encoder_mask_sample]
            dropped_section = encoder_grad_sample[np.logical_not(encoder_mask_sample)]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()
