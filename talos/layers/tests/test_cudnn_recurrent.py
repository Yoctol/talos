import pytest

import tensorflow as tf

from ..cudnn_recurrent import CuDNNGRU, CuDNNLSTM


@pytest.mark.parametrize('layer_cls', [CuDNNGRU, CuDNNLSTM])
def test_bypass_mask(layer_cls):
    layer = layer_cls(5, return_sequences=True)
    x = tf.zeros([3, 4, 5])
    x._keras_mask = tf.zeros([3, 4], dtype=tf.bool)
    out = layer(x)

    assert out._keras_mask is x._keras_mask
