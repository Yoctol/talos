import pytest

import tensorflow as tf

from ..sequential import Sequential


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_additional_inputs(graph):

    class LayerNeedSeqlen(tf.keras.layers.Layer):

        def call(self, x, seqlen):
            assert isinstance(seqlen, tf.Tensor)
            return x

        def compute_output_shape(self, input_shape):
            return input_shape

    sequential = Sequential([LayerNeedSeqlen(), tf.keras.layers.Dense(5)])
    inputs = tf.zeros([5, 4, 3], dtype=tf.float32)
    seqlen = tf.zeros([5], dtype=tf.int32)
    try:
        sequential(inputs, seqlen=seqlen)
    except Exception:
        pytest.fail("Failed to feed seqlen!")

    with pytest.raises(TypeError):
        sequential(inputs)


def test_additional_inputs_with_layer_accept_kwargs(graph):

    class LayerSupportKWArgs(tf.keras.layers.Layer):

        def call(self, x, **kwargs):
            assert 'training' in kwargs
            assert 'mask' in kwargs
            assert 'kwarg1' in kwargs
            assert 'kwarg2' in kwargs
            return x

        def compute_output_shape(self, input_shape):
            return input_shape

    sequential = Sequential([LayerSupportKWArgs(), tf.keras.layers.Dense(5)])
    inputs = tf.zeros([5, 4, 3], dtype=tf.float32)
    try:
        sequential(inputs, kwarg1=None, kwarg2=None)
    except Exception:
        pytest.fail("Failed to feed seqlen!")

    with pytest.raises(AssertionError):
        sequential(inputs)
