import pytest

import tensorflow as tf

from ..module import Sequential


@pytest.fixture(scope="function")
def graph():
    return tf.Graph()


def test_sequential(mocker, graph):
    with graph.as_default():
        mock_layer = mocker.Mock(
            side_effect=lambda x: x + 1,
            trainable_variables=[1, 2, 3],
            updates=[],
        )
        seq = Sequential([mock_layer for _ in range(5)])
    assert seq(2) == 7  # 2 + 1 * 5

    assert mock_layer.call_count == 5
    assert len(seq.trainable_variables) == 15
    assert len(seq.updates) == 0


def test_sequential_scope_name(mocker, graph):
    with graph.as_default():
        dense_layer1 = tf.layers.Dense(10)
        dense_layer2 = tf.layers.Dense(5)

        seq = Sequential([dense_layer1, dense_layer2], scope="test")

    assert seq(tf.constant([[1.], [2.]])).shape.as_list() == [2, 5]
    assert dense_layer1.kernel.name == "test/dense/kernel:0"
    assert dense_layer2.kernel.name == "test/dense_1/kernel:0"
