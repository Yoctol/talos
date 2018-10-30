import tensorflow as tf


_INITIALIZERS = {
    'lecun_normal': tf.keras.initializers.lecun_normal(),
    'orthogonal': tf.orthogonal_initializer(),
    'he_normal': tf.keras.initializers.he_normal(),
    'zero': tf.zeros_initializer(),
}


def get(initializer_id: str):
    try:
        initializer = _INITIALIZERS[initializer_id]
        return initializer
    except KeyError:
        raise KeyError(f"Unknown initializer_id: {initializer_id}")
