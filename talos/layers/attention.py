from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


class GlobalAttentionPooling1D(Layer):

    """Reference: https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(
            self,
            units,
            heads=1,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
        ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super().__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self.heads = int(heads)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_dim = input_shape[-1].value
        if input_dim is None:
            raise ValueError(
                'The last dimension of the inputs to `Dense` should be defined. Found `None`.',
            )
        self.input_spec = InputSpec(
            ndim=3,
            axes={-1: input_dim},
        )
        self.candidate_kernel = self.add_weight(
            'candidate_kernel',
            shape=[input_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.pre_softmax_kernel = self.add_weight(
            'pre_softmax_kernel',
            shape=[self.units, self.heads],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, seqlen=None):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        hidden_outputs = standard_ops.tensordot(inputs, self.candidate_kernel, [[2], [0]])
        if not context.executing_eagerly():
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.units]
            hidden_outputs.set_shape(output_shape)
        if self.activation is not None:
            hidden_outputs = self.activation(hidden_outputs)

        # shape (N, T, Head)
        logits = standard_ops.tensordot(hidden_outputs, self.pre_softmax_kernel, [[2], [0]])
        weights = nn.softmax(logits, axis=1)
        if seqlen is not None:
            mask = math_ops.to_float(array_ops.sequence_mask(seqlen))
            weights *= array_ops.expand_dims(mask, axis=-1)
            weights /= math_ops.reduce_sum(weights)

        # shape (N, Head, input_dim)
        outputs = standard_ops.matmul(weights, inputs, transpose_a=True)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape([input_shape[0], self.heads, input_shape[2]])

    def get_config(self):
        config = {
            'units': self.units,
            'heads': self.heads,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
