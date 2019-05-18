import re

import tensorflow as tf


_APPLY_MASK_MUL = 'apply_mask_mul'


def apply_mask(inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    if mask is None:
        return inputs
    if re.match(f".*{_APPLY_MASK_MUL}(_\d+)?", inputs.op.name):  # e.g. .../apply_mask_mul_1
        return inputs  # mask has already been applied

    i_rank = inputs.shape.ndims
    m_rank = mask.shape.ndims
    mask = tf.cast(mask, inputs.dtype)
    if i_rank > m_rank:
        indices = tuple(
            slice(None) if d < m_rank else tf.newaxis
            for d in range(i_rank)
        )  # expand multiple dims
        mask = mask[indices]
    elif i_rank < m_rank:
        raise ValueError(f"Invalid mask rank > inputs rank! {m_rank} > {i_rank}")

    return tf.multiply(inputs, mask, name=_APPLY_MASK_MUL)
