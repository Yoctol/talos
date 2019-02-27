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

        mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        mask_batch[:, 0] = True  # to make sure at least one True

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

    def test_forward_mask_gradients(self, inputs, sess):
        layer = TransformerBlock(units=3, heads=2, use_forward_mask=True)
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
        input_grads, encoder_grads = tf.gradients(outputs, [inputs, encoder_output])

        mask_batch = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
        encoder_mask_batch = np.random.choice(2, size=[5, maxlen_encoder]).astype(np.bool)
        mask_batch[:, 0] = True  # to make sure at least one True
        encoder_mask_batch[:, 0] = True

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

    def test_forward_mask_gradients(self, inputs, encoder_output, sess):
        layer = TransformerDecoderBlock(units=3, heads=2, use_forward_mask=True)
        maxlen, channel = inputs.shape.as_list()[1:]
        maxlen_encoder, channel_encoder = encoder_output.shape.as_list()[1:]

        outputs, _ = layer([inputs, encoder_output])
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
            feed_dict={
                inputs: np.random.rand(1, maxlen, channel),
                encoder_output: np.random.rand(1, maxlen_encoder, channel_encoder),
            },
        )
        for t, grad_of_output_t in enumerate(grad_list_val):
            attended_section = grad_of_output_t[:, :t + 1]
            dropped_section = grad_of_output_t[:, t + 1:]
            assert (attended_section != 0.).all()
            assert (dropped_section == 0.).all()
