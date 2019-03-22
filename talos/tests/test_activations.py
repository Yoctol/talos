import pytest

import tensorflow as tf

from ..activations import gelu


@pytest.mark.parametrize('function', [
    gelu,
])
def test_activations_registered_to_keras(function):
    assert callable(function)
    assert tf.keras.activations.get(f"talos.{function.__name__}") is function
