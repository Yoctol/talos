import tensorflow as tf

from ..positional_encode import PositionalEncode


def test_output_shape():
    layer = PositionalEncode()
    inputs = tf.zeros([5, 4, 3])
    outputs = layer(inputs)
    assert outputs.shape.as_list() == inputs.shape.as_list()
