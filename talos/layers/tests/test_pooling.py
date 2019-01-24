import pytest

import numpy as np
import tensorflow as tf

from talos.module import Sequential
from ..pooling import GlobalAttentionPooling1D
from ..pooling import GlobalAveragePooling1D


def test_global_attention_pooling_1d():
    width, channel = 10, 4
    units, heads = 3, 5
    att_pool = GlobalAttentionPooling1D(units=units, heads=heads)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel])
    outputs = att_pool(inputs)

    assert outputs.shape.as_list() == [None, heads, channel]
    assert att_pool.compute_output_shape(inputs.shape).as_list() == [None, heads, channel]
    assert len(att_pool.losses) == 0


def test_global_attention_pooling_1d_reuse():
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


def test_global_attention_pooling_1d_invalid_input_rank():
    att_pool = GlobalAttentionPooling1D(units=3, heads=4)
    rank2_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    with pytest.raises(ValueError):
        att_pool(rank2_inputs)

    rank4_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 5, 1])
    with pytest.raises(ValueError):
        att_pool(rank4_inputs)


def test_global_attention_pooling_1d_value():
    batch_size, maxlen = 3, 10
    att_pool = GlobalAttentionPooling1D(units=3, heads=4)
    inputs = tf.random_normal(shape=[batch_size, maxlen, 10])
    seqlen = tf.random_uniform(
        minval=3,
        maxval=maxlen,
        shape=[batch_size],
        dtype=tf.int32,
    )
    attended_vec = att_pool(inputs, seqlen=seqlen)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs_val, seqlen_val, attended_vec_val = sess.run(
            [inputs, seqlen, attended_vec])

    # since attend_vec is a weighted average of inputs
    for row, length, vec in zip(inputs_val, seqlen_val, attended_vec_val):
        min_val = np.min(row[:length], axis=0, keepdims=True)
        max_val = np.max(row[:length], axis=0, keepdims=True)
        assert np.logical_and(min_val < vec, vec < max_val).all()


def test_sequential_attention(graph):
    att_pool = GlobalAttentionPooling1D(units=3, heads=4)
    seq = Sequential([att_pool])
    inputs = tf.random_normal(shape=[1, 3, 10])
    seqlen = tf.random_uniform(
        minval=1,
        maxval=3,
        shape=[1],
        dtype=tf.int32,
    )
    seq(inputs, seqlen=seqlen)


def test_average_pooling_mask_value(graph):
    pool = GlobalAveragePooling1D()
    inputs = tf.placeholder(tf.float32, [None, 5, 1])
    seqlen = tf.placeholder(tf.int32, [None])
    outputs = pool(inputs, seqlen=seqlen)

    inputs_val = np.array([
        [[0], [1], [2], [3], [4]],
        [[2], [3], [4], [5], [6]],
    ], dtype=np.float32)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
        )
        outputs_val = sess.run(
            outputs,
            feed_dict={
                inputs: inputs_val,
                seqlen: [2, 4],
            },
        )

    expected_outputs_val = np.array([[0.5], [3.5]], dtype=np.float32)
    np.testing.assert_array_almost_equal(outputs_val, expected_outputs_val)


def test_given_mask(graph):
    att_pool = GlobalAttentionPooling1D(units=3, heads=4)
    pool = GlobalAveragePooling1D()
    inputs = tf.placeholder(tf.float32, [None, 5, 1])
    masked_inputs = tf.keras.layers.Masking()(inputs)

    att_pool(masked_inputs)
    pool(masked_inputs)
