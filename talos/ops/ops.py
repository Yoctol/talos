import tensorflow as tf


def sequence_reduce_mean(
        input_tensor: tf.Tensor,
        lengths: tf.Tensor,
    ):
    dtype = input_tensor.dtype
    maxlen = input_tensor.shape[1].value
    if maxlen is None:
        raise ValueError("'maxlen' should be known")
    seq_mask = tf.sequence_mask(lengths, maxlen=maxlen, dtype=dtype)

    masked_input = input_tensor * seq_mask
    masked_input_per_seq = tf.reduce_sum(masked_input, axis=1)
    mean_per_seq = masked_input_per_seq / tf.cast(lengths, dtype)
    mean_per_batch = tf.reduce_mean(mean_per_seq)
    return mean_per_batch
