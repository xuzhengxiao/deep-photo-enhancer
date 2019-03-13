import tensorflow as tf
from networks import D


def L2_loss(img1, img2, use_local_weight=None):
  if use_local_weight:
    w = -tf.log(tf.cast(img2, tf.float64) + tf.exp(tf.constant(-99, dtype=tf.float64))) + 1
    w = tf.cast(w * w, tf.float32)
    return tf.reduce_mean(w * tf.square(tf.subtract(img1, img2)))
  else:
    return tf.reduce_mean(tf.square(tf.subtract(img1, img2)))

def wgan_gp(fake_data, real_data):
  shape = tf.concat((tf.shape(real_data)[0:1], tf.tile([1], [real_data.shape.ndims - 1])), axis=0)
  alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
  interpolates = real_data + alpha * (fake_data - real_data)
  #interpolates = fake_data + alpha * (real_data - fake_data)
  interpolates.set_shape(real_data.get_shape().as_list())
  gradients = tf.gradients(D(interpolates),interpolates)[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
  #gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
  gradient_penalty=tf.reduce_mean(tf.maximum(0.,slopes-1.))
  return gradient_penalty
