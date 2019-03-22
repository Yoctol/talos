from typing import Callable, Tuple, Union

import tensorflow as tf

from talos.layers import Dropout, LayerNormalization
from talos.networks import Model

from .attention import MultiHeadSelfAttention, MultiHeadAttention


class TransformerBlock(Model):

    def __init__(
            self,
            units: int,
            heads: int,
            activation: Union[str, Callable] = 'relu',
            hidden_units: int = None,
            dropout_rate: float = 0.1,
            use_forward_mask: bool = False,
            heads_reg_coeff: float = None,
        ):
        """Reference: https://arxiv.org/abs/1706.03762
        """
        super().__init__()
        self.supports_masking = True

        self.units = units
        self.heads = heads
        self.use_forward_mask = use_forward_mask
        self.heads_reg_coeff = heads_reg_coeff

        if hidden_units is None:
            hidden_units = units * heads * 4  # ratio in paper
        self.hidden_dense = tf.keras.layers.Dense(
            units=hidden_units,
            activation=activation,
            use_bias=True,
        )
        self.ln_self_att, self.ln_ff = [LayerNormalization() for _ in range(2)]

        if 0. < dropout_rate < 1.:
            self.dropout_self_att, self.dropout_ff = [Dropout(dropout_rate) for _ in range(2)]
        else:
            self.dropout_self_att = self.dropout_ff = lambda x: x

        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        output_dim = input_shape[-1].value  # since the res-add-connection
        self.self_att = MultiHeadSelfAttention(
            units=self.units,
            heads=self.heads,
            output_dim=output_dim,
            use_forward_mask=self.use_forward_mask,
            heads_reg_coeff=self.heads_reg_coeff,
        )
        self.output_dense = tf.keras.layers.Dense(
            units=output_dim,
            use_bias=True,
        )
        self._input_spec.axes = {2: output_dim}  # since the res-add-connection
        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            mask: tf.Tensor = None,
            training: tf.Tensor = None,
        ) -> tf.Tensor:
        self_att_vec = self.dropout_self_att(
            self.self_att(self.ln_self_att(inputs), mask=mask),
            training=training,
        )

        outputs = self.dropout_ff(
            self.output_dense(
                self.hidden_dense(
                    self.ln_ff(self_att_vec + inputs),
                ),
            ),
            training=training,
        ) + self_att_vec

        if mask is not None:
            outputs *= tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask


class TransformerDecoderBlock(Model):

    def __init__(
            self,
            units: int,
            heads: int,
            activation: Union[str, Callable] = 'relu',
            hidden_units: int = None,
            dropout_rate: float = 0.1,
            use_forward_mask: bool = False,
            heads_reg_coeff: float = None,
        ):
        """Reference: https://arxiv.org/abs/1706.03762
        """
        super().__init__()
        self.supports_masking = True

        self.units = units
        self.heads = heads
        self.use_forward_mask = use_forward_mask
        self.heads_reg_coeff = heads_reg_coeff

        if hidden_units is None:
            hidden_units = units * heads * 4  # ratio in paper
        self.hidden_dense = tf.keras.layers.Dense(
            units=hidden_units,
            activation=activation,
            use_bias=True,
        )
        self.ln_self_att, self.ln_att, self.ln_ff = [LayerNormalization() for _ in range(3)]

        if 0. < dropout_rate < 1.:
            self.dropout_self_att, self.dropout_att, self.dropout_ff = [
                Dropout(dropout_rate)
                for _ in range(3)
            ]
        else:
            self.dropout_self_att = self.dropout_att = self.dropout_ff = lambda x: x

        self._input_spec = [tf.keras.layers.InputSpec(ndim=3) for _ in range(2)]

    @property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape_tuple):
        if len(input_shape_tuple) != 2:
            raise TypeError("both 'inputs' should be length 2 tuple!")

        input_shape, encoder_output_shape = input_shape_tuple

        output_dim = input_shape[-1].value  # since the res-add-connection
        self.self_att = MultiHeadSelfAttention(
            units=self.units,
            heads=self.heads,
            output_dim=output_dim,
            use_forward_mask=self.use_forward_mask,
            heads_reg_coeff=self.heads_reg_coeff,
        )
        self.encoder_decoder_att = MultiHeadAttention(
            units=self.units,
            heads=self.heads,
            output_dim=output_dim,
            heads_reg_coeff=self.heads_reg_coeff,
        )
        self.output_dense = tf.keras.layers.Dense(
            units=output_dim,
            use_bias=True,
        )
        self._input_spec[0].axes = {2: output_dim}  # since the res-add-connection
        super().build(input_shape_tuple)

    def call(
            self,
            inputs_tuple: Tuple[tf.Tensor, tf.Tensor],
            mask: Tuple[tf.Tensor, tf.Tensor] = None,
            training: tf.Tensor = None,
        ) -> tf.Tensor:
        if mask is None:
            mask = [None, None]
        if not (len(inputs_tuple) == len(mask) == 2):
            raise TypeError("both 'inputs' and 'mask' should be length 2 tuple!")

        inputs, encoder_outputs = inputs_tuple
        inputs_mask, encoder_outputs_mask = mask

        self_att_vec = self.dropout_self_att(
            self.self_att(
                self.ln_self_att(inputs),
                mask=inputs_mask,
            ),
            training=training,
        )

        att_vec = self.dropout_att(
            self.encoder_decoder_att(
                [self.ln_att(self_att_vec + inputs), encoder_outputs],
                mask=[inputs_mask, encoder_outputs_mask],
            ),
            training=training,
        )
        # since att will handle masking, multiply the res-part only.

        outputs = self.dropout_ff(
            self.output_dense(
                self.hidden_dense(
                    self.ln_ff(att_vec + self_att_vec),
                ),
            ),
            training=training,
        ) + att_vec

        if inputs_mask is not None:
            outputs *= tf.cast(inputs_mask, outputs.dtype)[:, :, tf.newaxis]

        # NOTE must be list!!!!! tuple will cause _set_mask_metadata wrong!!!!
        return [outputs, encoder_outputs]

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
