import pytest

import numpy as np
import tensorflow as tf

from ..layer_norm import LayerNormalization


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


@pytest.fixture
def layer():
    return LayerNormalization()


def test_output_shape(layer):
    shape = [3, 5]
    assert layer(tf.zeros(shape)).shape.as_list() == shape


def test_output_value(graph, layer):
    inputs = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    outputs = layer(inputs)

    with tf.Session() as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
        )
        np.testing.assert_array_almost_equal(
            sess.run(outputs),
            np.array([
                [-np.sqrt(1.5), 0., np.sqrt(1.5)],
                [-np.sqrt(1.5), 0., np.sqrt(1.5)],
            ]),
        )
