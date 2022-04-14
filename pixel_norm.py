import tensorflow as tensorflow

def keras_pixel_norm(x, epsilon = 1e-8):
    sqr = tf.keras.backend.square(x)
    mean_of_sqr = tf.keras.backend.mean(sqr, axis = 1, keepdims = True)
    rsqrt = 1 / tf.keras.backend.sqrt(mean_of_sqr + epsilon)
    return x * rsqrt
	
	
"""
Can be used with Lambda layer, e.g.:

inp = tf.keras.layers.Input((28, 28, 1))
conv = tf.keras.layers.Conv2D(16, 3, 2, 'same')(inp)
conv = tf.keras.layers.Lambda(keras_pixel_norm)(conv)
conv = tf.keras.layers.ReLU()(conv)

# and so on...
"""
