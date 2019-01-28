from tensorflow.python.keras.engine.sequential import Sequential as keras_Sequential
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.training.checkpointable import base as checkpointable


class Sequential(keras_Sequential):

    # HACK override: remove adding InputLayer part!!
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/engine/sequential.py#L122-L188
    # just copy/paste the source code and remove L142-L170
    @checkpointable.no_automatic_dependency_tracking
    def add(self, layer):
        # source code L137
        if not isinstance(layer, base_layer.Layer):
            raise TypeError(
                'The added layer must be '
                'an instance of class Layer. '
                f'Found: {layer}',
            )
        self.built = False

        # source code L172
        if self.outputs:
            # If the model is being built continuously on top of an input layer:
            # refresh its output.
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError(
                    'All layers in a Sequential model '
                    'should have a single output tensor. '
                    'For multi-output layers, '
                    'use the functional API.',
                )
                self.outputs = [output_tensor]

        self._layers.append(layer)
        if self._layers:
            self._track_layers(self._layers)
