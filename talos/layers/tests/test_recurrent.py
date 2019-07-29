import tensorflow as tf

from ..recurrent import GRUCell


def test_initial_state():
    cell = GRUCell(10)
    x = tf.zeros([3, 4])
    cell(x, cell.get_initial_state(x))
