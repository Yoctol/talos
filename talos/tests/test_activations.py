import pytest

import tensorflow as tf

from ..activations import gelu, gelu_v2


@pytest.mark.parametrize('function', [
    gelu,
    gelu_v2,
])
def test_activations_registered_to_keras(function):
    assert callable(function)
    assert tf.keras.activations.get(f"talos.{function.__name__}") is function
