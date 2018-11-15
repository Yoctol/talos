from tensorflow.python.eager import context
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.engine.sequential import Sequential as keras_Sequential
from tensorflow.python.util import tf_inspect


class Sequential(keras_Sequential):

    # override: add **kwargs
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/engine/sequential.py#L227-L233
    def call(self, inputs, training=None, mask=None, **kwargs):
        outputs, _ = self._call_and_compute_mask(
            inputs, training=training, mask=mask, **kwargs)
        return outputs

    # override: add **kwargs
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/engine/sequential.py#L235-L257
    def _call_and_compute_mask(self, inputs, training=None, mask=None, **kwargs):
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        for layer in self.layers:
            ### override part, other part copy/paste from source code. ###
            full_arg_spec = tf_inspect.getfullargspec(layer.call)
            if full_arg_spec.varkw is not None:
                # if layer.call supports **kwargs, feed everything to it
                needed_kwargs = {'training': training, 'mask': mask, **kwargs}
            else:
                # feed kwargs needed by layer.call only (kwargs & call_args)
                call_args = set(full_arg_spec.args)
                needed_kwargs = {
                    key: val for key, val in kwargs.items()
                    if key in call_args
                }
                if 'mask' in call_args:
                    needed_kwargs['mask'] = mask
                if 'training' in call_args:
                    needed_kwargs['training'] = training
            ##############################################################

            if isinstance(layer, Network) and layer._compute_output_and_mask_jointly:
                x, mask = layer._call_and_compute_mask(x, **needed_kwargs)
            else:
                x = layer.call(x, **needed_kwargs)
                if layer.supports_masking:
                    mask = layer.compute_mask(x, mask)
                else:
                    mask = None
            if not context.executing_eagerly():
                x._keras_mask = mask
        return x, mask
