from typing import List

import tensorflow as tf


class StackedRACells(tf.keras.layers.Layer):
    """Wrapper allowing a stack of RelativeAttention cells to behave as a single cell.

    Used to implement efficient stacked RelativeAttentionCells.

    Arguments:
        cells: List of RelativeAttention cell instances.

    Examples:
    ```python
        cells = [
            keras.layers.RelativeAttention(output_dim),
            keras.layers.RelativeAttention(output_dim),
            keras.layers.RelativeAttention(output_dim),
        ]
        inputs = keras.Input((timesteps, input_dim))
        x = keras.layers.StackedRelativeAttentionCells(cells)(inputs)
    ```
    """

    def __init__(self, cells, **kwargs):
        # todo validate all cells are RelativeAttentionCell ??
        self.cells = cells
        # if isinstance(cell, Layer):  # or RelativeAttentionCell
        super(StackedRACells, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        for cell in self.cells:
            cell.build(input_shape)
            input_shape = cell.compute_output_shape(input_shape)
        self._output_dim = input_shape[-1].value
        self.built = True

    def call(
            self,
            inputs: tf.Tensor,
            states: List[tf.Tensor] = None,
            mask: tf.Tensor = None,
            state_mask: tf.Tensor = None,
        ) -> tf.Tensor:

        if states is None:
            states = [None] * len(self.cells)

        if len(self.cells) != len(states):
            raise ValueError(f"The number of 'states' and 'cells' must be the same.")

        next_inputs = inputs
        new_states = []
        for cell, state in zip(self.cells, states):
            outputs, new_state, _, _ = cell.call(
                inputs=next_inputs,
                state=state,
                mask=mask,
                state_mask=state_mask,
            )
            assert new_state == next_inputs
            new_states.append(new_state)
            next_inputs = outputs
        return next_inputs, new_states, mask, mask

    @property
    def output_dim(self):
        return self._output_dim

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append({
                'class_name': cell.__class__.__name__,
                'config': cell.get_config(),
            })
        config = {'cells': cells}
        base_config = super(StackedRACells, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
