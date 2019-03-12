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

    def test_regularization(self, inputs, mask, layer_with_reg, sess):
        assert not layer_with_reg.losses
        layer_with_reg(inputs, mask=mask)

        assert len(layer_with_reg.losses) == 1
        assert layer_with_reg.losses[0].shape.ndims == 0

        maxlen, channel = inputs.shape.as_list()[1:]

        grad = tf.gradients(layer_with_reg.losses[0], inputs)[0]  # same shape as inputs

        mask_val = np.random.choice(2, size=maxlen).astype(np.bool)
        mask_val[:2] = True  # to make sure at least 2 True

        sess.run(tf.variables_initializer(var_list=layer_with_reg.variables))
        grad_val = sess.run(
            grad,
            feed_dict={
                inputs: [np.random.rand(maxlen, channel)],
                mask: [mask_val],
            },
        )[0]
        assert np.equal(
            grad_val != 0.,
            mask_val[:, np.newaxis],
        ).all()


class TestGlobalAttentionPooling1D(AttentionTestTemplate):

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAttentionPooling1D(units=3, heads=5)

    @pytest.fixture(scope='class')
    def layer_with_reg(self):
        return GlobalAttentionPooling1D(units=3, heads=5, heads_reg_coeff=0.01)

    def get_expected_shape(self, layer, inputs):
        return [inputs.shape[0].value, inputs.shape[2].value]


class TestMultiHeadSelfAttention(AttentionTestTemplate):

    @pytest.fixture(params=[
        MultiHeadSelfAttention(units=3, heads=2, output_dim=5),
        MultiHeadSelfAttention(units=3, heads=1, output_dim=5),
    ])
    def layer(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def layer_with_reg(self):
        return MultiHeadSelfAttention(units=3, heads=2, output_dim=5, heads_reg_coeff=0.01)

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
        grads_list = tf.stack([
            tf.gradients(outputs[:, t], inputs)[0]
            for t in range(maxlen)
        ])  # every elements have same shape as inputs

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grad_list_val = sess.run(
            grads_list,
            feed_dict={inputs: np.random.rand(5, maxlen, channel)},
        )
        assert np.equal(
            grad_list_val != 0.,
            np.tril(np.ones([maxlen, maxlen], dtype=np.bool))[:, np.newaxis, :, np.newaxis],
        ).all()


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
        inputs_grads, kv_grads = tf.gradients(outputs, [inputs, kv])

        inputs_mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        kv_mask_val = np.random.choice(2, size=[5, maxlen_encoder]).astype(np.bool)
        inputs_mask_val[:, :2] = True  # to make sure at least 2 True
        kv_mask_val[:, :2] = True

        sess.run(tf.variables_initializer(var_list=layer.variables))
        inputs_grads_val, kv_grads_val = sess.run(
            [inputs_grads, kv_grads],
            feed_dict={
                inputs: np.random.rand(5, maxlen, channel),
                kv: np.random.rand(5, maxlen_encoder, channel_encoder),
                mask: inputs_mask_val,
                kv_mask: kv_mask_val,
            },
        )
        assert np.equal(
            inputs_grads_val != 0.,
            inputs_mask_val[:, :, np.newaxis],
        ).all()
        assert np.equal(
            kv_grads_val != 0.,
            kv_mask_val[:, :, np.newaxis],
        ).all()
