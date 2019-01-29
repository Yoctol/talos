import tensorflow as tf

from ..attention import ScaledDotSelfAttention


def test_attention_output_shape():
    layer = ScaledDotSelfAttention(units=6, heads=2)
    inputs = tf.zeros([5, 4, 3])
    outputs = layer(inputs)
    assert outputs.shape.as_list() == [5, 4, 6 * 2]
    assert layer.compute_output_shape(inputs.shape) == [5, 4, 6 * 2]
