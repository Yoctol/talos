from tensorflow.python.eager import context
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.engine.sequential import Sequential as keras_Sequential
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import tf_inspect


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

    # HACK override: call compute_mask with input tensor not output
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/engine/sequential.py#L235-L257
    def _call_and_compute_mask(self, inputs, training=None, mask=None):
        if not self.built:
            self.build(inputs.shape)

        for layer in self.layers:
            kwargs = {}
            if 'mask' in tf_inspect.getfullargspec(layer.call).args:
                kwargs['mask'] = mask
            if 'training' in tf_inspect.getfullargspec(layer.call).args:
                kwargs['training'] = training

            if isinstance(layer, Network) and layer._compute_output_and_mask_jointly:
                outputs, mask = layer._call_and_compute_mask(inputs, **kwargs)
            else:
                outputs = layer.call(inputs, **kwargs)
                if layer.supports_masking:
                    mask = layer.compute_mask(inputs, mask)
                else:
                    mask = None
            if not context.executing_eagerly():
                outputs._keras_mask = mask
            inputs = outputs

        return outputs, mask

    # HACK override: fix output._keras_mask setting, and refactor nested block
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/engine/base_layer.py#L847-L868
    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        output_list = generic_utils.to_list(outputs)
        mask_already_computed = all(hasattr(x, '_keras_mask') for x in output_list)
        if hasattr(self, 'compute_mask') and not mask_already_computed:
            output_mask = self.compute_mask(inputs, previous_mask)
        else:
            # fix this line of source code
            output_mask = [x._keras_mask for x in output_list]

        if output_mask is None:
            output_mask_list = [None for _ in output_list]
        else:
            output_mask_list = generic_utils.to_list(output_mask)

        for x, m in zip(output_list, output_mask_list):
            try:
                x._keras_mask = m
            except AttributeError:
                pass
