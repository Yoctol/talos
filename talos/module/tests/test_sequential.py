import pytest

import tensorflow as tf

from ..sequential import Sequential


def test_build_sublayers_when_first_called(graph):
    sequential = Sequential([
        tf.keras.layers.Embedding(20, 10),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.Dense(5),
        tf.keras.layers.MaxPooling1D(),
    ])
    assert all(not layer.built for layer in sequential.layers)
    inputs = tf.zeros([1, 3], dtype=tf.float32)
    sequential(inputs)
    assert all(layer.built for layer in sequential.layers)


def test_context_manager_work_when_first_called(graph):
    sequential = Sequential([
        tf.keras.layers.Embedding(20, 10),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.Dense(5),
        tf.keras.layers.MaxPooling1D(),
    ])
    with tf.variable_scope('scope'):
        inputs = tf.zeros([1, 3], dtype=tf.float32)
        sequential(inputs)
    variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    assert all(var.graph is graph for var in variables)
    assert all(var.name.startswith('scope') for var in variables)


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
    # try: can successfully feed seqlen by keyword
    sequential(inputs, seqlen=seqlen)

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
    # try: can successfully feed kwargs
    sequential(inputs, kwarg1=None, kwarg2=None)

    with pytest.raises(AssertionError):
        sequential(inputs)
