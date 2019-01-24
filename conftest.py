import pytest

import tensorflow as tf


@pytest.yield_fixture(scope='function')
def graph():
    graph = tf.Graph()
    with graph.as_default():
        yield graph
