import abc

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine.training import Model as keras_Model


class Model(keras_Model, abc.ABC):

    # HACK override: remove pre-call part!!
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/network.py#L729-L814
    # just copy/paste the source code and remove L767-L802

    @base_layer.default
    def build(self, input_shape):
        # source code L752
        if self._is_graph_network:
            self.built = True
            return

        # If subclass network
        if input_shape is None:
            raise ValueError(
                'Input shape must be defined when calling build on a '
                'model subclass network.',
            )
        valid_types = (tuple, list, tensor_shape.TensorShape)
        if not isinstance(input_shape, valid_types):
            raise ValueError(
                'Specified input shape is not one of the valid types. '
                'Please specify a batch input shape of type tuple or '
                'list of input shapes. User provided '
                f'input type: {type(input_shape)}',
            )

        # remove L767-L802

        # source code L804
        if self._layers:
            self._track_layers(self._layers)
        if self.layers:
            for layer in self.layers:
                if not layer.built:
                    raise ValueError(
                        f'Layer: {layer} was not built in your model. Calling '
                        '`build` manually on a subclassed model is only '
                        'allowed for models with a static topology. '
                        'In this case, you can build your model by '
                        'calling it on real tensor data.',
                    )

        self.built = True

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass
