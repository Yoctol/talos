import tensorflow as tf


INITIALIZERS = {
    'lecun_normal': tf.keras.initializers.lecun_normal(),
    'orthogonal': tf.orthogonal_initializer(),
    'he_normal': tf.keras.initializers.he_normal(),
    'zero': tf.zeros_initializer(),
}
