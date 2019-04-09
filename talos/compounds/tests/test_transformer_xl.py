import pytest

import numpy as np
import tensorflow as tf

from ..transformer_xl import TransformerXL


@pytest.fixture(scope='module')
def layer():
    return TransformerXL(block_size=2, units=3, output_dim=10, heads=5)


def test_output_shape(layer, inputs):
    length = inputs.shape[1].value
    outputs = layer(inputs)
    assert outputs.shape.as_list() == [None, length, layer.output_dim]


def test_mask_gradients(inputs, mask, layer, sess):
    maxlen, channel = inputs.shape.as_list()[1:]

    outputs = layer(inputs, mask=mask)
    grads = tf.gradients(outputs, inputs)[0]  # same shape as inputs

    mask_val = np.random.choice(2, size=[5, maxlen]).astype(np.bool)
    mask_val[:, :2] = True  # to make sure at least 2 True

    sess.run(tf.variables_initializer(var_list=layer.variables))
    grads_val = sess.run(
        grads,
        feed_dict={
            inputs: np.random.rand(5, maxlen, channel),
            mask: mask_val,
        },
    )
    assert np.equal(
        grads_val != 0.,
        mask_val[:, :, np.newaxis],
    ).all()
