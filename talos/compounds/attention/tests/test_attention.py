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
        mask_batch[:, :2] = True  # to make sure at least 2 True

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
        layer = GlobalAttentionPooling1D(units=3, heads=4, heads_reg_coeff=1.0)

        assert not layer.losses
        layer(inputs)

        assert len(layer.losses) == 1
        assert layer.losses[0].shape.ndims == 0


class TestMultiHeadSelfAttention(AttentionTestTemplate):

    @pytest.fixture(params=[
        MultiHeadSelfAttention(units=3, heads=2, output_dim=5),
        MultiHeadSelfAttention(units=3, heads=1, output_dim=5),
    ])
    def layer(self, request):
        return request.param

    def get_expected_shape(self, layer, inputs):
        return inputs.shape.as_list()[:2] + [layer.output_dim]

    def test_regularization(self, inputs):
        layer = MultiHeadSelfAttention(units=3, heads=2, output_dim=5, heads_reg_coeff=0.1)

        assert not layer.losses
        layer(inputs)

        assert len(layer.losses) == 1
        assert layer.losses[0].shape.ndims == 0

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
    def kv(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, 5, 10])

    @pytest.fixture(scope='class')
    def kv_mask(self):
        return tf.placeholder(dtype=tf.bool, shape=[None, 5])

    @pytest.fixture(scope='class')
    def layer(self):
        return MultiHeadAttention(units=3, heads=2, output_dim=5)

    def test_multihead_attention(self, inputs, kv, layer):
        outputs = layer([inputs, kv])
        assert outputs.shape.as_list() == inputs.shape.as_list()[:2] + [layer.output_dim]

    def test_mask_gradients(self, inputs, mask, kv, kv_mask, layer, sess):
        maxlen, channel = inputs.shape.as_list()[1:]
        maxlen_encoder, channel_encoder = kv.shape.as_list()[1:]

        outputs = layer(
            [inputs, kv],
            mask=[mask, kv_mask],
        )
        input_grads, kv_grads = tf.gradients(outputs, [inputs, kv])

        mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        kv_mask_batch = np.random.choice(2, size=[5, maxlen_encoder]).astype(np.bool)
        mask_batch[:, :2] = True  # to make sure at least 2 True
        kv_mask_batch[:, :2] = True

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grads_batch, kv_grads_batch = sess.run(
            [input_grads, kv_grads],
            feed_dict={
                inputs: [np.random.rand(maxlen, channel) for _ in range(5)],
                kv: [
                    np.random.rand(maxlen_encoder, channel_encoder)
                    for _ in range(5)
                ],
                mask: mask_batch,
                kv_mask: kv_mask_batch,
            },
        )

        for mask_sample, grad_sample, kv_mask_sample, kv_grad_sample in zip(
                mask_batch,
                grads_batch,
                kv_mask_batch,
                kv_grads_batch,
            ):
            attended_section = grad_sample[mask_sample]
            dropped_section = grad_sample[np.logical_not(mask_sample)]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()

            attended_section = kv_grad_sample[kv_mask_sample]
            dropped_section = kv_grad_sample[np.logical_not(kv_mask_sample)]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()
