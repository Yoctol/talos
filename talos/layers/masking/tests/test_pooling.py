import pytest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from ..pooling import MaskGlobalAveragePooling1D


class TestMaskGlobalAveragePooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(tf.float32, [None, 5, 1])

    @pytest.fixture(scope='class')
    def mask(self, inputs):
        maxlen = inputs.shape[1].value
        return tf.placeholder(tf.bool, [None, maxlen])

    @pytest.fixture(scope='class')
    def layer(self):
        return MaskGlobalAveragePooling1D()

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
