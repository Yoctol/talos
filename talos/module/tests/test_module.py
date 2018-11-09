import pytest

import tensorflow as tf

from ..module import Sequential


@pytest.yield_fixture(scope="function")
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph


def test_logic_by_mock(mocker):
    mock_layer = mocker.Mock(
        side_effect=lambda x: x + 1,
        trainable_variables=[1, 2, 3],
        updates=['a', 'b'],
    )
    n_layers = 5
    seq = Sequential([mock_layer for _ in range(n_layers)])
    assert seq(2) == 2 + n_layers * 1

    assert mock_layer.call_count == n_layers
    assert len(seq.trainable_variables) == 3 * n_layers
    assert len(seq.updates) == 2 * n_layers


def test_layer_build(graph):
    batch_size, input_dim = None, 4
    dense1 = tf.keras.layers.Dense(10)
    dense2 = tf.keras.layers.Dense(5)
    bn_layer = tf.keras.layers.BatchNormalization()
    seq = Sequential([dense1, dense2, bn_layer], scope="scope")

    inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim])
    outputs = seq(inputs)

    assert outputs.shape.as_list() == [batch_size, dense2.units]
    assert dense1.kernel.shape.as_list() == [input_dim, dense1.units]
    assert dense2.kernel.shape.as_list() == [dense1.units, dense2.units]
    assert dense1.kernel.name == "scope/dense/kernel:0"  # test scope works
    assert dense2.kernel.name == "scope/dense_1/kernel:0"
    assert len(seq.trainable_variables) == 6  # 2 dense layers' kernel/bias + bn gamma/beta
    assert len(seq.non_trainable_variables) == 2  # sliding average mean, variance of bn
    assert len(seq.variables) == 6 + 2
    assert len(seq.updates) == 2  # sliding average mean, variance update ops of bn


def test_reuse(graph):
    dense1 = tf.keras.layers.Dense(10)
    dense2 = tf.keras.layers.Dense(5)
    seq = Sequential([dense1, dense2])

    rank2_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    rank3_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 5])
    rank4_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 2, 5])
    invalid_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    seq(rank2_inputs)  # built when first call, will check input_dim = 5 at future call
    try:
        seq(rank3_inputs)
        seq(rank4_inputs)
    except Exception:
        pytest.fail("Sequential can't reuse!")

    with pytest.raises(ValueError):
        seq(invalid_inputs)  # due to last dimension incompatible: 2 != 5


def test_backprop(graph):
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

    assert gradients_norm_val > 0.  # gradient can flow to input side.


def test_nested_sequential(graph):
    seq = Sequential(sub_layers=[
        Sequential(sub_layers=[tf.keras.layers.Dense(units) for _ in range(3)], scope=f'B{i}')
        for i, units in enumerate([1, 2, 3], 1)
    ], scope='A')
    inputs = tf.random_normal(shape=[5, 3])
    seq(inputs)

    assert len(seq.variables) == 18  # 9 dense layers' kernel/bias
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B1')) == 6
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B2')) == 6
    assert len(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A/B3')) == 6
