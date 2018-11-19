import tensorflow as tf


class Embedding(tf.keras.layers.Embedding):

    def __init__(
            self,
            input_dim,
            output_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_index=None,
            input_length=None,
            **kwargs,
        ):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', tf.float32)
        super(tf.keras.layers.Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

        self._valid_mask_index(mask_index)
        self.mask_index = mask_index
        self.supports_masking = (mask_index is not None)
        self.input_length = input_length

    def _valid_mask_index(self, mask_index):
        if mask_index is None:
            return
        if not isinstance(mask_index, int):
            raise ValueError("`mask_index` should be integer!")
        if not (0 <= mask_index < self.input_dim):
            raise ValueError("`mask_index` should be in range [0, input_dim)!")

    def compute_mask(self, inputs, mask=None):
        if self.mask_index is None:
            return None

        return tf.not_equal(inputs, self.mask_index)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
            'mask_index': self.mask_index,
            'input_length': self.input_length,
        }
        return config
