import pytest

import numpy as np
import tensorflow as tf

from ..pooling import GlobalAttentionPooling1D
from ..pooling import GlobalAveragePooling1D


class TestGlobalPooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(tf.float32, [None, 5, 1])

    @pytest.fixture(scope='class')
    def seqlen(self):
        return tf.placeholder(tf.int32, [None])

    @pytest.fixture(scope='class')
    def layer(self):
        return GlobalAveragePooling1D()

    def test_mask_value(self, sess, inputs, seqlen, layer):
        outputs = layer(inputs, seqlen=seqlen)

        inputs_val = np.array([
            [0., 1., 2., 3., 4.],
            [2., 3., 4., 5., 6.],
        ]).reshape(-1, 5, 1)

        sess.run(tf.variables_initializer(var_list=layer.variables))
        outputs_val = sess.run(
            outputs,
            feed_dict={
                inputs: inputs_val,
                seqlen: [2, 4],
            },
        )

        expected_outputs_val = np.array([(0. + 1.) / 2., (2. + 3. + 4. + 5.) / 4.]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(outputs_val, expected_outputs_val)

    def test_support_masked_inputs(self, inputs, layer):
        masked_inputs = tf.keras.layers.Masking()(inputs)
        layer(masked_inputs)


class TestGlobalAttentionPooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, 4, 3])

    @pytest.fixture(scope='class')
    def seqlen(self):
        return tf.placeholder(dtype=tf.int32, shape=[None])

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

    def test_output_value(self, sess, inputs, seqlen):
        maxlen, channel = inputs.shape.as_list()[1:]
        # zero init to isolate the grad through weights calculation.
        att_pool = GlobalAttentionPooling1D(units=3, heads=4, kernel_initializer='zeros')
        attended_vec = att_pool(inputs, seqlen=seqlen)
        grads = tf.gradients(attended_vec, inputs)[0]  # same shape as inputs

        seqlen_batch = np.random.randint(2, maxlen + 1, size=[5])
        sess.run(tf.variables_initializer(var_list=att_pool.variables))
        grad_batch = sess.run(
            grads,
            feed_dict={
                inputs: [np.random.rand(maxlen, channel) for _ in seqlen_batch],
                seqlen: seqlen_batch,
            },
        )
        for seqlen_sample, grad_sample in zip(seqlen_batch, grad_batch):
            attended_section = grad_sample[:seqlen_sample]
            assert np.logical_and(0. < attended_section, attended_section < att_pool.heads).all()
            assert (grad_sample[seqlen_sample:] == 0.).all()

    def test_support_masked_inputs(self, inputs, layer):
        masked_inputs = tf.keras.layers.Masking()(inputs)
        layer(masked_inputs)
