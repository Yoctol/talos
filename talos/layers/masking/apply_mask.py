import tensorflow as tf


class ApplyMask(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            i_rank = inputs.shape.ndims
            m_rank = mask.shape.ndims
            if i_rank > m_rank:
                indices = (slice(None),) * m_rank + (tf.newaxis,) * (i_rank - m_rank)
                inputs *= mask[indices]
            elif i_rank == m_rank:
                inputs *= mask
            else:
                raise ValueError(f"Invalid mask rank > inputs rank! {m_rank} > {i_rank}")

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        return mask
