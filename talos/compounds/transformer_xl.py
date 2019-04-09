from typing import Callable, Union

import tensorflow as tf

from talos.layers import Dropout, LayerNormalization
from talos.networks import Model

from .attention import RelativeAttentionCell


class TransformerXL(Model):

    def __init__(
            self,
            block_size: int,
            units: int,
            heads: int,
            activation: Union[str, Callable] = 'relu',
            hidden_units: int = None,
            dropout_rate: float = 0.1,
            use_forward_mask: bool = False,
        ):
        super().__init__()
        self.supports_masking = True

        if hidden_units is None:
            hidden_units = units * heads * 4  # ratio in paper
        self.hidden_dense = tf.keras.layers.Dense(
            units=hidden_units,
            activation=activation,
            use_bias=True,
        )
        self.ln_pre_cell, self.ln_pre_ff = [LayerNormalization() for _ in range(2)]

        if 0. < dropout_rate < 1.:
            self.dropout_cell, self.dropout_ff = [Dropout(dropout_rate) for _ in range(2)]
        else:
            self.dropout_cell = self.dropout_ff = lambda x: x

        self.block_size = block_size
        self.units = units
        self.heads = heads
        self.use_forward_mask = use_forward_mask
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        output_dim = input_shape[-1].value
        self.cell = RelativeAttentionCell(
            units=self.units,
            output_dim=output_dim,
            heads=self.heads,
            use_forward_mask=self.use_forward_mask,
        )
        self.output_dense = tf.keras.layers.Dense(
            units=output_dim,
            use_bias=True,
        )
        self._input_spec.axes = {2: output_dim}
        super().build(input_shape)

    def call(self, inputs, mask: tf.Tensor = None, training: tf.Tensor = None):
        ln_inputs = self.ln_pre_cell(inputs)

        maxlen = inputs.shape[1].value
        block_size_list = [self.block_size for _ in range(maxlen // self.block_size)]
        if maxlen % self.block_size != 0:
            block_size_list.append(maxlen % self.block_size)
        block_input_list = tf.split(ln_inputs, block_size_list, axis=1)

        if mask is not None:
            block_mask_list = tf.split(mask, block_size_list, axis=1)
        else:
            block_mask_list = [None for _ in block_input_list]

        state = None
        state_mask = None
        output_list = []
        for block_input, block_mask in zip(block_input_list, block_mask_list):
            block_output = self.cell(
                block_input,
                state=state,
                mask=block_mask,
                state_mask=state_mask,
            )
            output_list.append(block_output)
            state = block_input
            state_mask = block_mask

        att_vec = tf.concat(output_list, axis=1)
        att_vec = self.dropout_cell(att_vec, training=training)

        outputs = self.ln_pre_ff(att_vec + inputs)
        outputs = self.hidden_dense(outputs)
        outputs = self.output_dense(outputs)
        outputs = self.dropout_ff(outputs, training=training) + att_vec

        if mask is not None:
            outputs *= tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
