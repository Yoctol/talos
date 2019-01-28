import pytest

import numpy as np
import tensorflow as tf

from ..model import Model
from ..sequential import Sequential


class ModelSupportMask(Model):

    def __init__(self):
        super().__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return inputs * tf.cast(mask, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


@pytest.mark.parametrize('network', [
    ModelSupportMask(),
    Sequential([ModelSupportMask()]),
])
def test_support_mask_args(sess, network):
    outputs = network(
        tf.constant([1., 2., 3., 4., 5.]),
        mask=tf.constant([True, False, True, True, False]),
    )
    np.testing.assert_array_almost_equal(
        sess.run(outputs),
        np.array([1., 0., 3., 4., 0.]),
    )
