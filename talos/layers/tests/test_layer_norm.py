import pytest

import numpy as np
import tensorflow as tf

from ..layer_norm import LayerNormalization


@pytest.fixture
def layer():
    return LayerNormalization()


def test_output_shape(layer):
    shape = [3, 5]
    assert layer(tf.zeros(shape)).shape.as_list() == shape


def test_output_value(layer):
    inputs = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    outputs = layer(inputs)

    with tf.Session() as sess:
        sess.run(tf.variables_initializer(var_list=layer.variables))
        np.testing.assert_array_almost_equal(
            sess.run(outputs),
            np.array([
                [-np.sqrt(1.5), 0., np.sqrt(1.5)],
                [-np.sqrt(1.5), 0., np.sqrt(1.5)],
            ]),
        )
