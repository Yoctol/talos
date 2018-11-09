import pytest

import tensorflow as tf

from ..module import Sequential


@pytest.yield_fixture(scope="function")
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_logic(mocker):
    mock_layer = mocker.Mock(
        side_effect=lambda x: x + 1,
        trainable_variables=[1, 2, 3],
        updates=['a', 'b'],
    )
    seq = Sequential([mock_layer for _ in range(5)])
    assert seq(2) == 7  # 2 + 1 * 5

    assert mock_layer.call_count == 5
    assert len(seq.trainable_variables) == 15
    assert len(seq.updates) == 10


def test_layer_build(graph):
    dense_layer1 = tf.keras.layers.Dense(10)
    dense_layer2 = tf.keras.layers.Dense(5)
    bn_layer = tf.keras.layers.BatchNormalization()
    seq = Sequential([dense_layer1, dense_layer2, bn_layer], scope="test")

    inputs = tf.constant([[1.], [2.]])
    outputs = seq(inputs)

    assert outputs.shape.as_list() == [2, 5]
    assert dense_layer1.kernel.shape.as_list() == [1, 10]
    assert dense_layer2.kernel.shape.as_list() == [10, 5]
    assert dense_layer1.kernel.name == "test/dense/kernel:0"
    assert dense_layer2.kernel.name == "test/dense_1/kernel:0"
    assert len(seq.trainable_variables) == 6  # dense weight/bias + bn gamma/beta
    assert len(seq.non_trainable_variables) == 2  # sliding average mean, variance
    assert len(seq.variables) == 6 + 2
    assert len(seq.updates) == 2  # sliding average mean, variance


def test_reuse(graph):
    dense_layer1 = tf.keras.layers.Dense(10)
    dense_layer2 = tf.keras.layers.Dense(5)
    seq = Sequential([dense_layer1, dense_layer2])

    rank2_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    rank3_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 5])
    rank4_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 2, 5])
    invalid_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    seq(rank2_inputs)
    try:
        seq(rank3_inputs)
        seq(rank4_inputs)
    except Exception:
        pytest.fail("Sequential can't reuse!")

    with pytest.raises(ValueError):
        seq(invalid_inputs)


def test_gradients(graph):
    seq = Sequential([tf.keras.layers.Dense(units) for units in range(3, 10)])
    inputs = tf.random_normal(shape=[5, 3])
    outputs = seq(inputs)
    gradients = tf.gradients(outputs, inputs)
    gradients_norm = tf.norm(gradients)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
        )
        gradients_norm_val = sess.run(gradients_norm)

    assert gradients_norm_val > 0.


def test_nested_sequential(graph):
    seq = Sequential(sub_layers=[
        Sequential(sub_layers=[tf.keras.layers.Dense(units) for _ in range(3)], scope=f'B{i}')
        for i, units in enumerate([1, 2, 3], 1)
    ], scope='A')
    inputs = tf.random_normal(shape=[5, 3])
    seq(inputs)

    assert len(seq.variables) == 18  # 9 dense layers
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B1')) == 6
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B2')) == 6
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B3')) == 6
