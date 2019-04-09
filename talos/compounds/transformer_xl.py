import tensorflow as tf

from talos.layers import Layer
from .attention import RelativeAttentionCell


class TransformerXL(Layer):

    def __init__(
            self,
            block_size: int,
            units: int,
            output_dim: int,
            heads: int = 1,
            use_forward_mask: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.cell = RelativeAttentionCell(
            units=units,
            output_dim=output_dim,
            heads=heads,
            use_forward_mask=use_forward_mask,
        )
        self.block_size = block_size
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property
    def units(self):
        return self.cell.units

    @property
    def output_dim(self):
        return self.cell.output_dim

    @property
    def heads(self):
        return self.cell.heads

    @property
    def trainable_weights(self):
        return self.cell.trainable_weights

    def call(self, inputs, mask=None):
        maxlen = inputs.shape[1].value
        block_size_list = [self.block_size for _ in range(maxlen // self.block_size)]
        if maxlen % self.block_size != 0:
            block_size_list.append(maxlen % self.block_size)
        block_input_list = tf.split(inputs, block_size_list, axis=1)

        if mask is not None:
            block_mask_list = tf.split(mask, block_size_list, axis=1)
        else:
            block_mask_list = [None for _ in block_input_list]

        state = None
        state_mask = None
        output_list = []
        for block_input, mask in zip(block_input_list, block_mask_list):
            block_output = self.cell(block_input, state=state, mask=mask, state_mask=state_mask)
            output_list.append(block_output)
            state = block_input
            state_mask = mask

        output = tf.concat(output_list, axis=1)
        return output
