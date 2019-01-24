import gc
import pytest

import tensorflow as tf


@pytest.fixture
def graph():
    tf.reset_default_graph()
    yield tf.get_default_graph()
    gc.collect()
