import tensorflow as tf

init_w = 1e-3
WEIGHTS_INIT_STDEV = .01


def _fc_layer(inputs, output_size, name):
  shape = inputs.shape
  dim = 1
  for d in shape[1:]:
    dim *= d.value
  x = tf.reshape(inputs, [-1, dim])

  weight_shape = [dim, output_size]
  bias_shape = [output_size]

  weights = tf.get_variable(name + 'weight', weight_shape, dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=WEIGHTS_INIT_STDEV, seed=1))
  biases = tf.get_variable(name + 'bias', bias_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

  # x_drop = tf.nn.dropout(x, keep_prob=0.7)
  fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
  return fc


def _conv_init_vars(net, out_channels, filter_size, name, transpose=False, trainable=True):
  _, rows, cols, in_channels = [i.value for i in net.get_shape()]
  if not transpose:
    weights_shape = [filter_size, filter_size, in_channels, out_channels]
  else:
    weights_shape = [filter_size, filter_size, out_channels, in_channels]

  initializer = tf.contrib.layers.variance_scaling_initializer(init_w, seed=1)
  weights_init = tf.get_variable(name, weights_shape, dtype=tf.float32,
                                 initializer=initializer, trainable=trainable)
  return weights_init


def _conv_layer(net, num_filters, filter_size, strides, name, trainable=True, padding="VALID", pad_mode="SYMMETRIC"):
  weights_init = _conv_init_vars(net, num_filters, filter_size, name + 'kernel', trainable=trainable)
  strides_shape = [1, strides, strides, 1]
  pad_size = (filter_size - 1) // 2
  if pad_size > 0 and pad_mode is not None:
    net = tf.pad(net, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], pad_mode)
  net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)
  return net


def _conv_block(net, num_filters, filter_sizer, strides, name, trainable=True, padding="VALID", pad_mode="SYMMETRIC",conv_first=True,
                **kwargs):
  _name=name
  if conv_first:
    net = _conv_layer(net, num_filters, filter_sizer, strides, name, trainable, padding, pad_mode)
  act = {"lrelu": tf.nn.leaky_relu, "selu": selu}
  if "nonlinearity" in kwargs:
    assert kwargs["nonlinearity"] in act, "nolinearity should be lrelu or selu"
    net = act[kwargs["nonlinearity"]](net)
  if "norm_type" in kwargs:
    if kwargs["norm_type"] == "batch":
      if "norm_name" in kwargs:
        name = name + kwargs["norm_name"]
    net = _batch_norm(net, name, trainable, kwargs["norm_type"])
  if not conv_first:
    net = _conv_layer(net, num_filters, filter_sizer, strides, _name, trainable, padding, pad_mode)
  return net


# def _conv_tranpose_layer(net, num_filters, filter_size, strides, name,norm="in1", trainable=True):
#   weights_init = _conv_init_vars(net, num_filters, filter_size, name + 'kernel', transpose=True, trainable=trainable)
#
#   batch_size = tf.shape(net)[0]
#   rows = tf.shape(net)[1]
#   cols = tf.shape(net)[2]
#   new_rows, new_cols = rows * strides, cols * strides
#   # new_shape = #tf.stack([tf.shape(net)[0], new_rows, new_cols, num_filters])
#
#   new_shape = [batch_size, new_rows, new_cols, num_filters]
#   tf_shape = tf.stack(new_shape)
#   strides_shape = [1, strides, strides, 1]
#
#   net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
#   net = _batch_norm(net, name + norm, trainable)
#   net = selu(net)
#   return net


"""according to bn_name to assign different value in initializaion"""
def _batch_norm(net, name, trainable=True, norm_type="batch", channels=None):
  if channels:
    var_shape = [channels]
  else:
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
  if norm_type == "batch":
    mu, sigma_sq = tf.nn.moments(net, [0, 1, 2], keep_dims=True)
  else:  # instance
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
  shift = tf.get_variable(name + 'shift', var_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                          trainable=trainable)
  scale = tf.get_variable(name + 'scale', var_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0),
                          trainable=trainable)
  epsilon = 1e-5  # 1e-3
  normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
  return scale * normalized + shift


def selu(x):
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def _spacial_repeat(img, shape_like):
  height = shape_like[1]
  width = shape_like[2]
  multiples = tf.stack([1, height, width, 1])
  img = tf.tile(img, multiples=multiples)
  return img


def resize_layer(tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
  t_shape = tensor.get_shape().as_list()
  if t_shape[1] == None or t_shape[2] == None:
    t_shape = tf.shape(tensor)
  t_size = [t_shape[1] * 2, t_shape[2] * 2]
  tensor = tf.image.resize_images(tensor, t_size, method=method)
  return tensor


def resize_concat(tensor, concat_tensor, name, num_filters, trainable, **kwargs):
  deconv = resize_layer(tensor)
  deconv = tf.concat([deconv, concat_tensor], -1)
  deconv = _conv_block(deconv, num_filters, 3, 1, name, trainable=trainable,conv_first=False, **kwargs)
  return deconv


# U-Net
def G(img, trainable=True, norm_name="bn1"):
  with tf.variable_scope("generator"):
    kwargs = {"nonlinearity": "selu", "norm_type": "batch", "norm_name": norm_name}
    conv1 = _conv_block(img, 16, 3, 1, "conv1", trainable=trainable, **kwargs)  # /1  512->512
    conv2 = _conv_block(conv1, 32, 5, 2, "conv2", trainable=trainable, **kwargs)  # /2  512->256
    conv3 = _conv_block(conv2, 64, 5, 2, "conv3", trainable=trainable, **kwargs)  # /4  256->128
    conv4 = _conv_block(conv3, 128, 5, 2, "conv4", trainable=trainable, **kwargs)  # /8  128->64
    conv5 = _conv_block(conv4, 128, 5, 2, "conv5", trainable=trainable, **kwargs)  # /16  64->32  [-1,32,32,128]

    # global feature
    conv6 = _conv_block(conv5, 128, 5, 2, "conv6", trainable=trainable, **kwargs)  # /32  32->16
    conv7 = _conv_block(conv6, 128, 5, 2, "conv7", trainable=trainable, **kwargs)  # /64  16->8

    global_feature = _conv_block(conv7, 128, 8, 1, "global_feature_1", trainable=trainable, pad_mode=None,
                                 **kwargs)  # [-1,1,1,128]
    global_feature = _conv_block(global_feature, 128, 1, 1, "global_feature_2", trainable=trainable, pad_mode=None,
                                 **kwargs)  # [-1,1,1,128]

    global_feature = _spacial_repeat(global_feature, [1, 32, 32, 1])  # [-1,32,32,128]

    middle = tf.concat([conv5, global_feature], -1)
    # only conv layer
    middle = _conv_block(middle, 128, 3, 1, "middle", trainable=trainable)  # [-1,32,32,128]


    deconv4 = resize_concat(middle, conv4, "deconv5", 128, trainable)
    deconv3 = resize_concat(deconv4, conv3, "deconv4", 64, trainable)
    deconv2 = resize_concat(deconv3, conv2, "deconv3", 32, trainable)
    deconv1 = resize_concat(deconv2, conv1, "deconv2", 16, trainable)

    uconv1 = _conv_block(deconv1, 16, 3, 1, "uconv1", trainable=trainable,conv_first=False, **kwargs)
    uconv0 = _conv_block(uconv1, 3, 3, 1, "uconv0", trainable=trainable,conv_first=False, **kwargs)
    res = img + uconv0

    # res=tf.clip_by_value(res,0,1)
    return res


def D(img, trainable=True):
  with tf.variable_scope("discriminator"):
    kwargs = {"nonlinearity": "lrelu", "norm_type": "batch"}
    #kwargs = {"nonlinearity": "lrelu", "norm_type": "batch"}
    conv1 = _conv_block(img, 16, 3, 1, "conv1", trainable=trainable, **kwargs)  # /1  512->512
    conv2 = _conv_block(conv1, 32, 5, 2, "conv2", trainable=trainable, **kwargs)  # /2  512->256
    conv3 = _conv_block(conv2, 64, 5, 2, "conv3", trainable=trainable, **kwargs)  # /4  256->128
    conv4 = _conv_block(conv3, 128, 5, 2, "conv4", trainable=trainable, **kwargs)  # /8  128->64
    conv5 = _conv_block(conv4, 128, 5, 2, "conv5", trainable=trainable, **kwargs)  # /16  64->32
    conv6 = _conv_block(conv5, 128, 5, 2, "conv6", trainable=trainable, **kwargs)  # /16  32->16   [-1,16,16,128]
    # conv7 = _conv_layer(conv6, 1, 16, 1, "conv7", trainable=trainable, pad_mode=None)
    # res = tf.reduce_mean(conv7, [1, 2, 3])
    res = tf.reduce_mean(conv6, [2, 3])
    res=_fc_layer(res,1,"fc")
    return res
