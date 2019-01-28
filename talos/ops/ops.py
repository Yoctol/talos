import tensorflow as tf


def masked_reduce_mean(
        inputs: tf.Tensor,  # shape (N, T)
        mask: tf.Tensor,  # shape (N, T)
        true_count: tf.Tensor = None,
    ):
    mask = tf.cast(mask, inputs.dtype)

    masked_input = mask * inputs  # shape (N, T)
    sum_per_seq = tf.reduce_sum(masked_input, axis=1)  # shape (N, )

    if true_count is None:  # use pre computed true_count
        true_count = tf.reduce_sum(mask, axis=1)  # shape (N, )
    else:
        true_count = tf.cast(true_count, inputs.dtype)

    mean_per_seq = sum_per_seq / (true_count + tf.keras.backend.epsilon())
    mean_per_batch = tf.reduce_mean(mean_per_seq)
    return mean_per_batch


def sequence_reduce_mean(
        inputs: tf.Tensor,
        lengths: tf.Tensor,
    ):
    maxlen = inputs.shape[1].value
    seq_mask = tf.sequence_mask(lengths, maxlen=maxlen, dtype=inputs.dtype)

    return masked_reduce_mean(
        inputs,
        mask=seq_mask,
        true_count=lengths,
    )
