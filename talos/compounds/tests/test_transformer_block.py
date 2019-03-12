import pytest

import numpy as np
import tensorflow as tf

from ..transformer_block import TransformerBlock, TransformerDecoderBlock


class TestTransformerBlock:

    @pytest.fixture(scope='class')
    def layer(self):
        return TransformerBlock(units=3, heads=2)

    def test_output_shape(self, inputs, layer):
        outputs = layer(inputs)
        assert outputs.shape.as_list() == inputs.shape.as_list()

    def test_masked_inputs_propagate(self, masked_inputs, layer):
        outputs = layer(masked_inputs)
        assert outputs._keras_mask is masked_inputs._keras_mask

    def test_mask_gradients(self, inputs, mask, layer, sess):
        maxlen, channel = inputs.shape.as_list()[1:]

        outputs = layer(inputs, mask=mask)
        grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

        mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        mask_val[:, :2] = True  # to make sure at least one True

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grad_val = sess.run(
            grads,
            feed_dict={
                inputs: np.random.rand(5, maxlen, channel),
                mask: mask_val,
            },
        )
        assert np.equal(
            grad_val != 0.,
            mask_val[:, :, np.newaxis],
        ).all()

    def test_forward_mask_gradients(self, inputs, sess):
        layer = TransformerBlock(units=3, heads=2, use_forward_mask=True)
        maxlen, channel = inputs.shape.as_list()[1:]

        outputs = layer(inputs)
        grads_list = tf.stack([
            tf.gradients(outputs[:, t], inputs)[0]
            for t in range(maxlen)
        ])  # every elements have same shape as inputs

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grad_list_val = sess.run(
            grads_list,
            feed_dict={inputs: np.random.rand(1, maxlen, channel)},
        )
        assert np.equal(
            grad_list_val != 0.,
            np.tril(np.ones([maxlen, maxlen], dtype=np.bool))[:, np.newaxis, :, np.newaxis],
        ).all()


class TestTransformerDecoderBlock:

    @pytest.fixture(scope='class')
    def layer(self):
        return TransformerDecoderBlock(units=3, heads=2)

    @pytest.fixture(scope='class')
    def encoder_output(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, 5, 10])

    @pytest.fixture(scope='class')
    def encoder_output_mask(self):
        return tf.placeholder(dtype=tf.bool, shape=[None, 5])

    def test_output_shape(self, inputs, encoder_output, layer):
        output_list = layer([inputs, encoder_output])
        assert output_list[0].shape.as_list() == inputs.shape.as_list()
        assert output_list[1] is encoder_output

    def test_masked_inputs_propagate(self, masked_inputs, encoder_output, layer):
        output_list = layer([masked_inputs, encoder_output])
        assert output_list[0]._keras_mask is masked_inputs._keras_mask
        assert output_list[1] is encoder_output

    def test_mask_gradients(self, inputs, mask, encoder_output, encoder_output_mask, layer, sess):
        maxlen, channel = inputs.shape.as_list()[1:]
        maxlen_encoder, channel_encoder = encoder_output.shape.as_list()[1:]

        outputs, _ = layer(
            [inputs, encoder_output],
            mask=[mask, encoder_output_mask],
        )
        inputs_grads, encoder_grads = tf.gradients(outputs, [inputs, encoder_output])

        inputs_mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        encoder_mask_val = np.random.choice(2, size=[5, maxlen_encoder]).astype(np.bool)
        inputs_mask_val[:, :2] = True  # to make sure at least 2 True
        encoder_mask_val[:, :2] = True

        sess.run(tf.variables_initializer(var_list=layer.variables))
        inputs_grads_val, encoder_grads_val = sess.run(
            [inputs_grads, encoder_grads],
            feed_dict={
                inputs: np.random.rand(5, maxlen, channel),
                encoder_output: np.random.rand(5, maxlen_encoder, channel_encoder),
                mask: inputs_mask_val,
                encoder_output_mask: encoder_mask_val,
            },
        )
        assert np.equal(
            inputs_grads_val != 0.,
            inputs_mask_val[:, :, np.newaxis],
        ).all()
        assert np.equal(
            encoder_grads_val != 0.,
            encoder_mask_val[:, :, np.newaxis],
        ).all()

    def test_forward_mask_gradients(self, inputs, encoder_output, sess):
        layer = TransformerDecoderBlock(units=3, heads=2, use_forward_mask=True)
        maxlen, channel = inputs.shape.as_list()[1:]
        maxlen_encoder, channel_encoder = encoder_output.shape.as_list()[1:]

        outputs, _ = layer([inputs, encoder_output])
        grads_list = tf.stack([
            tf.gradients(outputs[:, t], inputs)[0]
            for t in range(maxlen)
        ])  # every elements have same shape as inputs

        sess.run(tf.variables_initializer(var_list=layer.variables))
        grad_list_val = sess.run(
            grads_list,
            feed_dict={
                inputs: np.random.rand(1, maxlen, channel),
                encoder_output: np.random.rand(1, maxlen_encoder, channel_encoder),
            },
        )
        assert np.equal(
            grad_list_val != 0.,
            np.tril(np.ones([maxlen, maxlen], dtype=np.bool))[:, np.newaxis, :, np.newaxis],
        ).all()
