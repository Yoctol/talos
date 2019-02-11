import tensorflow as tf

from talos.layers import Dropout, LayerNormalization
from talos.networks import Model

from .attention import ScaledDotSelfAttention


class TransformerBlock(Model):

    def __init__(
            self,
            units: int,
            heads: int,
            hidden_units: int = None,
            dropout_rate: float = 0.1,
        ):
        """Reference: https://arxiv.org/abs/1706.03762
        """
        super().__init__()
        self.supports_masking = True

        self.units = units
        self.heads = heads

        if hidden_units is None:
            hidden_units = units * heads * 4  # ratio in paper
        self.hidden_dense = tf.keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            use_bias=True,
        )
        self.ln_att = LayerNormalization()
        self.ln_ff = LayerNormalization()
        self.dropout_layer = Dropout(dropout_rate)
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        output_dim = input_shape[-1].value  # since the res-add-connection
        self.att = ScaledDotSelfAttention(
            units=self.units, heads=self.heads, output_dim=output_dim)
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
        normed_inputs = self.ln_att(inputs)
        att_vec = self.dropout_layer(
            self.att(normed_inputs, mask=mask),
            training=training,
        )
        # since att will handle masking, multiply the res-part only.
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)
            normed_att_vec = self.ln_ff(att_vec + inputs * mask[:, :, tf.newaxis])
        else:
            normed_att_vec = self.ln_ff(att_vec + inputs)

        outputs = self.dropout_layer(
            self.output_dense(self.hidden_dense(normed_att_vec)),
            training=training,
        ) + att_vec
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask
