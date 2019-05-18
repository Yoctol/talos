import tensorflow as tf


def apply_mask(inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    if mask is not None:
        i_rank = inputs.shape.ndims
        m_rank = mask.shape.ndims
        casted_mask = tf.cast(mask, inputs.dtype)
        if i_rank > m_rank:
            indices = tuple(
                slice(None) if d < m_rank else tf.newaxis
                for d in range(i_rank)
            )  # expand multiple dims
            inputs *= casted_mask[indices]
        elif i_rank == m_rank:
            inputs *= casted_mask
        else:
            raise ValueError(f"Invalid mask rank > inputs rank! {m_rank} > {i_rank}")

    return inputs
