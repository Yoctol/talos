import tensorflow as tf


class GlobalAttentionPooling1D(tf.keras.layers.Layer):
    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(
            self,
            units,
            heads=1,
            activation='tanh',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            reg_coeff=0.,
            **kwargs,
        ):
        super().__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.units = units
        self.heads = heads
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if reg_coeff < 0:
            raise ValueError("reg_coeff can't be negative!")
        self.reg_coeff = reg_coeff
        self._identity_matrix = None
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        self.candidate_kernel = self.add_weight(
            name='candidate_kernel',
            shape=(input_shape[2].value, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.softmax_kernel = self.add_weight(
            name='softmax_kernel',
            shape=(self.units, self.heads),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='candidate_bias',
                shape=(self.units, ),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            seqlen: tf.Tensor = None,
        ) -> tf.Tensor:
        # shape (N, T, units)
        hidden_outputs = tf.tensordot(inputs, self.candidate_kernel, axes=[[2], [0]])
        if self.use_bias:
            hidden_outputs = tf.nn.bias_add(hidden_outputs, self.bias)
        if self.activation is not None:
            hidden_outputs = self.activation(hidden_outputs)
        # shape (N, T, Head)
        logits = tf.tensordot(hidden_outputs, self.softmax_kernel, axes=[[2], [0]])
        weights = tf.nn.softmax(logits, axis=1)
        if seqlen is not None:
            # Renormalize for lower seqlen
            maxlen = inputs.shape[1].value
            # shape (N, T)
            mask = tf.sequence_mask(seqlen, maxlen=maxlen, dtype=self.dtype)
            weights *= tf.expand_dims(mask, axis=2)
            weights /= tf.reduce_sum(weights, axis=1, keepdims=True)

        if self.reg_coeff > 0:
            weights_product = tf.matmul(weights, weights, transpose_a=True)
            identity_matrix = self._get_identity_matrix()
            penalty = self.reg_coeff * tf.reduce_sum(tf.square(
                weights_product - identity_matrix,
            ))
            self.add_loss(penalty)
        # shape (N, Head, input_dim)
        outputs = tf.matmul(weights, inputs, transpose_a=True)
        return outputs

    def _get_identity_matrix(self):
        if self._identity_matrix is None:
            self._identity_matrix = tf.eye(self.heads, batch_shape=[1], dtype=self.dtype)
        return self._identity_matrix

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[1] = self.heads
        return tf.TensorShape(output_shape)
