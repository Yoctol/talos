import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(
            self,
            center: bool = True,
            scale: bool = True,
            trainable: bool = True,
            axis: int = -1,
            **kwargs,
        ):
        self.center = center
        self.scale = scale
        self.trainable = trainable
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=input_shape[self.axis],
                initializer='zeros',
                trainable=self.trainable,
            )
        else:
            self.beta = None

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=input_shape[self.axis],
                initializer='ones',
                trainable=self.trainable,
            )
        else:
            self.gamma = None

        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[self.axis], keep_dims=True)
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
