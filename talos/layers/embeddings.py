from typing import Sequence, Union

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils import tf_utils


class Embedding(tf.keras.layers.Embedding):

    def __init__(
            self,
            vocab_size,
            embeddings_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_index: Union[int, Sequence[int]] = None,
            input_length: int = None,
            **kwargs,
        ):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', tf.float32)
        super(tf.keras.layers.Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = self.vocab_size = vocab_size
        self.output_dim = self.embeddings_dim = embeddings_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

        self.mask_index = self._standardize_mask_index(mask_index)
        self.supports_masking = (mask_index is not None)
        self.input_length = input_length
        self.auxiliary_token = 0
        self._constant = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self._constant:
            self.embeddings = tf.constant(self.embeddings_initializer.value)
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                trainable=self.trainable,
            )
        if self.auxiliary_token > 0:
            # HACK, since Layer.add_weight will take
            # the intersection of trainable (in arg) and self.trainable
            # manually set self.trainable = True
            # to make sure auxiliary_embeddings is tracked by backend.
            original_trainable = self.trainable
            self.trainable = True
            self.auxiliary_embeddings = self.add_weight(
                shape=(self.auxiliary_token, self.output_dim),
                name='auxiliary_embeddings',
                trainable=True,
            )
            self.trainable = original_trainable
            self.total_embeddings = tf.concat(
                [self.embeddings, self.auxiliary_embeddings],
                axis=0,
                name='total_embeddings',
            )
        else:
            self.total_embeddings = self.embeddings
        self.built = True

    @property
    def trainable_weights(self):
        # HACK in keras implementation, they consider layer.trainable as well,
        # be it's ignored in this part.
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        # HACK in keras implementation, they consider layer.trainable as well,
        # be it's ignored in this part.
        return self._non_trainable_weights

    @classmethod
    def from_weights(
            cls,
            weights: np.ndarray,
            mask_index: Union[int, Sequence[int]] = None,
            constant: bool = False,
            auxiliary_token: int = 0,
            **kwargs,
        ):
        if weights.ndim != 2:
            raise ValueError(f"`weights` should be a rank 2 array! Recieved shape: {weights.shape}")
        vocab_size, embeddings_dim = weights.shape
        initializer = tf.constant_initializer(weights)
        layer = cls(
            vocab_size=vocab_size,
            embeddings_dim=embeddings_dim,
            embeddings_initializer=initializer,
            mask_index=mask_index,
            **kwargs,
        )
        if constant:
            layer.trainable = False
            layer._constant = True
        layer.auxiliary_token = auxiliary_token
        return layer

    def _standardize_mask_index(self, mask_index):
        if mask_index is None:
            return None
        if isinstance(mask_index, Sequence):
            for idx in mask_index:
                self._valid_mask_index_int(idx)
            return set(mask_index)

        self._valid_mask_index_int(mask_index)
        return mask_index

    def _valid_mask_index_int(self, mask_index):
        if not isinstance(mask_index, int):
            raise ValueError("`mask_index` should be integer!")
        if not (0 <= mask_index < self.input_dim):
            raise ValueError("`mask_index` should be in range [0, input_dim)!")

    def call(self, inputs, mask=None):
        if inputs.dtype not in (tf.int32, tf.int64):
            inputs = tf.cast(inputs, tf.int32)

        out = tf.nn.embedding_lookup(self.total_embeddings, inputs)
        return out

    def compute_mask(self, inputs, mask):
        if self.mask_index is None:
            return mask

        if isinstance(self.mask_index, int):
            new_mask = tf.not_equal(inputs, self.mask_index)
        else:
            new_mask = tf.reduce_all(
                [tf.not_equal(inputs, idx) for idx in self.mask_index],
                axis=0,
            )

        if mask is None:
            return new_mask
        return tf.logical_and(mask, new_mask)

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'embeddings_dim': self.embeddings_dim,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
            'mask_index': self.mask_index,
            'input_length': self.input_length,
            'auxiliary_token': self.auxiliary_token,
        }
        return config
