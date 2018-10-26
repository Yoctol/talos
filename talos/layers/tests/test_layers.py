import tensorflow as tf

from ..layers import Dense


def test_dense():
    dense_layer = Dense(units=10)
    assert dense_layer.units == 10
    x = tf.placeholder(dtype=tf.float32, shape=[None, 20])
    y = dense_layer(x)
    assert y.shape.as_list() == [None, 10]
