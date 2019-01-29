import pytest

import numpy as np
import tensorflow as tf

from ..positional_encode import PositionalEncode


@pytest.fixture(scope='module')
def layer():
    return PositionalEncode()


def test_output_shape(layer):
    inputs = tf.zeros([5, 4, 3])
    outputs = layer(inputs)
    assert outputs.shape.as_list() == inputs.shape.as_list()


def test_output_val(layer, sess):
    inputs = tf.constant([
        [[1., 2., 3., 4.]],
        [[5., 6., 7., 8.]],
    ])
    outputs = layer(inputs)
    inputs_val, outputs_val = sess.run([inputs, outputs])
    np.testing.assert_array_almost_equal(
        outputs_val[:, 0],
        inputs_val[:, 0] + np.array([
            [0., 1., 0., 1.]  # sin0, cos0, sin0,
            for _ in range(len(inputs_val))
        ]),
    )
