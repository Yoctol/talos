import abc
from typing import List
from itertools import chain

import tensorflow as tf


class Module(abc.ABC):

    def __init__(
            self,
            sub_layers: List,
            scope: str = None,
        ):
        self.sub_layers = sub_layers
        self.scope = scope

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if self.scope is not None:
            with tf.variable_scope(self.scope):
                output = self.call(x)
        else:
            output = self.call(x)
        return output

    @abc.abstractmethod
    def call(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @property
    def trainable_variables(self):
        return list(chain.from_iterable(
            layer.trainable_variables for layer in self.sub_layers)
        )

    @property
    def updates(self):
        return list(chain.from_iterable(
            layer.updates for layer in self.sub_layers)
        )


class Sequential(Module):

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.sub_layers:
            x = layer(x)
        return x
