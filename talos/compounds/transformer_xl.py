from typing import Callable, Union

import tensorflow as tf

from talos.layers import Dropout, LayerNormalization
from talos.networks import Model

from .attention import RelativeAttentionCell


class TransformerXL(Model):

    """Transformer-XL: Zihang Dai

    ref: https://arxiv.org/pdf/1901.02860.pdf

    Args:
        block_size (int): The length of segment (L)
        units (int): The dimensionality of feature space
            in each attention cell.
        heads (int): The number of heads for attention (h).
        activation (str or callable): Activation for hidden position-wise feed forward layer.
            Default: relu.
        hidden_units (int): Defaults to `units * heads * 4`.
        dropout_rate (float): Defaults to 0.1.
        use_forward_mask (bool): Whether to mask out future information. Defaults to False.

    """

    def __init__(
            self,
            block_size: int,
            units: int,
            heads: int,
            state_gradient: bool = False,
            bidirectional: bool = False,
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
            self.dropout_cell = self.dropout_ff = lambda x, training: x

        self.block_size = block_size
        self.units = units
        self.heads = heads
        self.state_gradient = state_gradient
        self.bidirectional = bidirectional
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
        if self.bidirectional:
            self.backward_cell = RelativeAttentionCell(
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
        """
        Args:
            inputs: a float tensor with shape (batch_size, full_timesteps, input_dim).
            mask: a boolean tensor with shape (batch_size, full_timesteps).
                  to mask out the information of outputs.
            training: a boolen tensor. Defaults to None.

        Return:
            a float tensor with shape (batch_size, full_timesteps, output_dim).
        """
        # RelationAttention SubLayers
        ln_inputs = self.ln_pre_cell(inputs)  # layer norm
        att_vec = self._blockwise_attention(ln_inputs, mask=mask, cell=self.cell)
        if self.bidirectional:
            if mask is not None:
                backward_mask = tf.reverse(mask, axis=[1])
            else:
                backward_mask = None
            backward_vec = tf.reverse(
                self._blockwise_attention(
                    tf.reverse(ln_inputs, axis=[1]),
                    mask=backward_mask,
                    cell=self.backward_cell,
                ),
                axis=[1],
            )
            att_vec = att_vec + backward_vec

        att_vec = self.dropout_cell(att_vec, training=training)

        # Position-wise Feed Forward
        outputs = self.ln_pre_ff(att_vec + inputs)  # layer norm
        outputs = self.hidden_dense(outputs)  # dense layer(hidden units)
        outputs = self.output_dense(outputs)  # dense layer(output_dim)
        outputs = self.dropout_ff(outputs, training=training) + att_vec  # res-connect

        if mask is not None:
            outputs *= tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]
        return outputs

    def _blockwise_attention(self, inputs, mask, cell):
        # split full_timesteps to blocks as possible
        # with 1 < length <= self.block_size
        maxlen = inputs.shape[1].value
        block_size_list = [self.block_size for _ in range(maxlen // self.block_size)]
        if maxlen % self.block_size != 0:
            block_size_list.append(maxlen % self.block_size)

        if len(block_size_list) > 1:
            block_input_list = tf.split(inputs, block_size_list, axis=1)
        else:
            block_input_list = [inputs]

        if mask is not None:
            if len(block_size_list) > 1:
                block_mask_list = tf.split(mask, block_size_list, axis=1)
            else:
                block_input_list = [mask]
        else:
            block_mask_list = [None for _ in block_input_list]

        state = None
        state_mask = None
        output_list = []
        for block_input, block_mask in zip(block_input_list, block_mask_list):
            if block_mask is not None:
                block_output = tf.cond(
                    tf.reduce_any(block_mask),
                    lambda: cell(
                        block_input,
                        state=state,
                        mask=block_mask,
                        state_mask=state_mask,
                    ),
                    lambda: tf.zeros_like(block_input),
                )
            else:
                block_output = cell(block_input, state=state)

            output_list.append(block_output)
            state = block_input
            if not self.state_gradient:
                state = tf.stop_gradient(state)
            state_mask = block_mask

        if len(output_list) > 1:
            att_vec = tf.concat(output_list, axis=1)
        else:
            att_vec = output_list[0]
        return att_vec

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
