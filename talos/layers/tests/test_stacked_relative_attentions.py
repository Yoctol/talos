import pytest

import tensorflow as tf

from ..stacked_relative_attentions import StackedRACells
from ...compounds.attention.recursive import RelativeAttentionCell


@pytest.fixture(scope='module')
def params():
    return [
        {"units": 3, "output_dim": 10, "heads": 5},
        {"units": 5, "output_dim": 20, "heads": 7},
        {"units": 7, "output_dim": 30, "heads": 9},
    ]


@pytest.fixture(scope='module')
def cells(params):
    return [RelativeAttentionCell(**param) for param in params]


@pytest.fixture(scope='module')
def stacked_cells(cells):
    return StackedRACells(cells)


@pytest.fixture(scope='module')
def states(inputs, params):
    batch_size, length, init_dim = inputs.shape.as_list()
    different_length = length + 3
    states = [
        tf.placeholder(
            dtype=inputs.dtype,
            shape=[batch_size, different_length, init_dim],
        ),
    ]
    for param in params[:-1]:
        states.append(
            tf.placeholder(
                dtype=inputs.dtype,
                shape=[batch_size, different_length, param['output_dim']],
            ),
        )
    return states


# @pytest.fixture(scope='module')
# def state_masks(mask, states):
#     return [
#         tf.placeholder(
#             dtype=mask.dtype,
#             shape=state.shape.as_list()[:-1],
#         )
#         for state in states
#     ]


def test_output_shape(stacked_cells, inputs):
    length = inputs.shape[1].value
    outputs, _, _, _ = stacked_cells(inputs, None)
    assert outputs.shape.as_list() == [None, length, stacked_cells.output_dim]


def test_shape_of_output_states(stacked_cells, cells, inputs):
    length = inputs.shape[1].value
    init_dim = inputs.shape[2].value
    _, states, _, _ = stacked_cells(inputs, None)

    # get output dims
    output_dims = []
    outputs = inputs
    for cell in cells:
        outputs, _, _, _ = cell(outputs)
        output_dims.append(cell.output_dim)

    for state, output_dim in zip(states, [init_dim, *output_dims][:-1]):
        assert state.shape.as_list() == [None, length, output_dim]


def test_call_with_states(stacked_cells, states, inputs):
    outputs, o_states, o_mask, o_mask_state = stacked_cells(inputs, states)
