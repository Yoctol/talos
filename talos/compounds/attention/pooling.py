import tensorflow as tf

from talos.networks import Model


_LARGE_BIAS = 1e4


class GlobalAttentionPooling1D(Model):
    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(
            self,
            units: int,
            heads: int = 1,
            activation: str = 'tanh',
            use_bias: bool = False,
            heads_reg_coeff: float = None,
            **kwargs,
        ):
        super().__init__()
        self.units = units
        self.heads = heads
        self.activation = activation
        self.use_bias = use_bias

        self.hidden_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
        )
        self.presoftmax_layer = tf.keras.layers.Dense(units=heads)

        if heads_reg_coeff is not None and heads_reg_coeff < 0:
            raise ValueError("reg_coeff can't be negative!")
        self.heads_reg_coeff = heads_reg_coeff

        self.supports_masking = True
        self._input_spec = tf.keras.layers.InputSpec(ndim=3)

    @property  # override property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        output_dim = input_shape[-1].value
        self.output_layer = tf.keras.layers.Dense(units=output_dim)
        self._input_spec.axes = {2: output_dim}
        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            mask: tf.Tensor = None,
        ) -> tf.Tensor:
        # shape (N, T, units)
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)

        hidden_outputs = self.hidden_layer(inputs)  # shape (N, T, U)
        logits = self.presoftmax_layer(hidden_outputs)  # shape (N, T, h)
        logits = self._mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, axis=1)  # shape (N, T, h)

        # shape (N, h, input_dim)
        attended_vec = tf.matmul(weights, inputs, transpose_a=True)

        if self.heads_reg_coeff is not None and self.heads > 1:
            # NOTE add input loss more than once
            # may cause dependencies error
            if self.inputs:
                raise RuntimeError(
                    "Layer with inputs related regularization "
                    "should not be called more than once!",
                )
            reg_loss = self._disagreement_output_loss(attended_vec)
            self.output_layer.add_loss(self.heads_reg_coeff * reg_loss)

        inputs_dim = inputs.shape[-1].value
        flatten_attended_vec = tf.reshape(attended_vec, [-1, self.heads * inputs_dim])
        outputs = self.output_layer(flatten_attended_vec)
        return outputs

    def _mask_logits(self, logits, mask):
        if mask is not None:
            bias = (1. - mask) * _LARGE_BIAS  # shape (N, T)
            logits -= bias[:, :, tf.newaxis]

        return logits

    def _disagreement_output_loss(self, attended_vec):
        unit_head_vec = tf.nn.l2_normalize(
            attended_vec,
            axis=-1,
        )  # shape (N, h, U)
        cosine_similarity = tf.matmul(
            unit_head_vec,
            unit_head_vec,
            transpose_b=True,
        )  # shape (N, h, h)
        return tf.reduce_mean(cosine_similarity)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], input_shape[2]])

    def compute_mask(self, inputs, mask):
        return None
