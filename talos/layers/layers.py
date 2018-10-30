from typing import Tuple

import tensorflow as tf

from .. import activations
from .. import initializers


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
        activation = activations.get(activation)
        kernel_initializer = initializers.get(kernel_initializer)
        bias_initializer = initializers.get(bias_initializer)

        return tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            trainable=trainable,
            name=name,
        )


class Conv1D:

    def __new__(
            filters: int,
            kernel_size: int,
            strides: int = 1,
            padding: str = 'valid',
            data_format: str = 'channels_last',
            dilation_rate: int = 1,
            activation: str = 'linear',
            use_bias: bool = True,
            kernel_initializer: str = 'lecun_normal',
            bias_initializer: str = 'zero',
            trainable: bool = True,
            name: str = None,
        ):
        activation = activations.get(activation)
        kernel_initializer = initializers.get(kernel_initializer)
        bias_initializer = initializers.get(bias_initializer)

        return tf.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            trainable=trainable,
            name=name,
        )


class Conv2D:

    def __new__(
            filters: int,
            kernel_size: int,
            strides: int = (1, 1),
            padding: str = 'valid',
            data_format: str = 'channels_last',
            dilation_rate: int = 1,
            activation: str = 'linear',
            use_bias: bool = True,
            kernel_initializer: str = 'lecun_normal',
            bias_initializer: str = 'zero',
            trainable: bool = True,
            name: str = None,
        ):
        activation = activations.get(activation)
        kernel_initializer = initializers.get(kernel_initializer)
        bias_initializer = initializers.get(bias_initializer)

        return tf.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            trainable=trainable,
            name=name,
        )


class Conv2DTranspose:

    def __new__(
            filters: int,
            kernel_size: int,
            strides: Tuple[int, int] = (1, 1),
            padding: str = 'valid',
            data_format: str = 'channels_last',
            dilation_rate: int = 1,
            activation: str = 'linear',
            use_bias: bool = True,
            kernel_initializer: str = 'lecun_normal',
            bias_initializer: str = 'zero',
            trainable: bool = True,
            name: str = None,
        ):
        activation = activations.get(activation)
        kernel_initializer = initializers.get(kernel_initializer)
        bias_initializer = initializers.get(bias_initializer)

        return tf.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            trainable=trainable,
            name=name,
        )
