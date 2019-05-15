import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class Dropout(tf.keras.layers.Dropout):

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return tf.nn.dropout(
                inputs,
                rate=self.rate,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
            )
        output = tf_utils.smart_cond(
            training,
            dropped_inputs,
            lambda: array_ops.identity(inputs),
        )
        return output
