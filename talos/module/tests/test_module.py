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

    seq = Sequential([dense_layer1, dense_layer2], scope="test")
    inputs = tf.constant([[1.], [2.]])
    outputs = seq(inputs)

    assert outputs.shape.as_list() == [2, 5]
    assert dense_layer1.kernel.shape.as_list() == [1, 10]
    assert dense_layer2.kernel.shape.as_list() == [10, 5]
    assert dense_layer1.kernel.name == "test/dense/kernel:0"
    assert dense_layer2.kernel.name == "test/dense_1/kernel:0"
    assert len(seq.variables) == 4
    assert len(seq.trainable_variables) == 4
    assert len(seq.non_trainable_variables) == 0


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
