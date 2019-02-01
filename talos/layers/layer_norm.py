import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(
            self,
            center: bool = True,
            scale: bool = True,
            beta_initializer: str = 'zeros',
            gamma_initializer: str = 'ones',
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if (self.center or self.scale) and input_shape[-1].value is None:
            raise ValueError(
                f'The last dimension of the inputs to `LayerNormalization` '
                'should be defined. Found `None`.',
            )
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=[input_shape[-1].value],
                initializer=self.beta_initializer,
            )
        else:
            self.beta = None

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=[input_shape[-1].value],
                initializer=self.gamma_initializer,
            )
        else:
            self.gamma = None

        if self.center or self.scale:
            self.input_spec = tf.keras.layers.InputSpec(axes={
                -1: input_shape[-1].value,
            })

        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        return tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=tf.keras.backend.epsilon(),
        )

    def compute_output_shape(self, input_shape):
        return input_shape
