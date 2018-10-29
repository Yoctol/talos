import tensorflow as tf

from ..activations import ACTIVATIONS
from ..initializers import INITIALIZERS


class Dense:

    def __new__(
            cls,
            units,
            activation: str = 'linear',
            use_bias: bool = True,
            kernel_initializer: str = 'lecun_normal',
            bias_initializer: str = 'zero',
            trainable: bool = True,
            name: str = None,
        ):
        activation = ACTIVATIONS[activation]
        kernel_initializer = INITIALIZERS[kernel_initializer]
        bias_initializer = INITIALIZERS[bias_initializer]

        return tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            trainable=trainable,
            name=name,
        )
