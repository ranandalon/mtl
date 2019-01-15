import tensorflow as tf

def conv(x, receptive_field_shape, channels_shape, stride, name, repad=False):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[-1]]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))

  if repad:
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
    conv = tf.nn.conv2d(padded, weights, strides=[1, stride, stride, 1], padding='VALID')
  else:
    conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')

  conv_bias = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(tf.contrib.layers.batch_norm(conv_bias))

def deconv(x, receptive_field_shape, channels_shape, stride, name):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[0]]

  input_shape = x.get_shape().as_list()
  batch_size = input_shape[0]
  height = input_shape[1]
  width = input_shape[2]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))
  conv = tf.nn.conv2d_transpose(x, weights, [batch_size, height * stride, width * stride, channels_shape[0]], [1, stride, stride, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(tf.contrib.layers.batch_norm(conv_bias))

def deconvNoRelu(x, receptive_field_shape, channels_shape, stride, name):
  kernel_shape = receptive_field_shape + channels_shape
  bias_shape = [channels_shape[0]]

  input_shape = x.get_shape().as_list()
  batch_size = input_shape[0]
  height = input_shape[1]
  width = input_shape[2]

  weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
  biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))
  conv = tf.nn.conv2d_transpose(x, weights, [batch_size, height * stride, width * stride, channels_shape[0]], [1, stride, stride, 1], padding='SAME')
  conv_bias = tf.nn.bias_add(conv, biases)
  return conv_bias


def max_pool(x, size, stride, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding, name='maxpool')

def max_pool_argmax(x, size, stride, padding='SAME'):
  return tf.nn.max_pool_with_argmax(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding, name='maxpoolArgmax')

def unpool(x, size):
  out = tf.concat_v2([x, tf.zeros_like(x)], 3)
  out = tf.concat_v2([out, tf.zeros_like(out)], 2)

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * size, sh[2] * size, sh[3]]
    return tf.reshape(out, out_size)

  shv = tf.shape(x)
  ret = tf.reshape(out, tf.stack([-1, shv[1] * size, shv[2] * size, sh[3]]))
  ret.set_shape([None, None, None, sh[3]])
  return ret

def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2]*shape[3]),
                   argmax % (shape[2]*shape[3]) // shape[3]]
    return tf.stack(output_list)

def unpool_layer2x2_batch(bottom, argmax):
  bottom_shape = tf.shape(bottom)
  top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]
  # top_shape = tf.stack(bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3])

  batch_size = top_shape[0]
  height = top_shape[1]
  width = top_shape[2]
  channels = top_shape[3]

  argmax_shape = tf.to_int64([batch_size, height, width, channels])
  argmax = unravel_argmax(argmax, argmax_shape)

  t1 = tf.to_int64(tf.range(channels))
  t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
  t1 = tf.reshape(t1, [-1, channels])
  t1 = tf.transpose(t1, perm=[1, 0])
  t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
  t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

  t2 = tf.to_int64(tf.range(batch_size))
  t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
  t2 = tf.reshape(t2, [-1, batch_size])
  t2 = tf.transpose(t2, perm=[1, 0])
  t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

  t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

  t = tf.concat([t2, t3, t1], 4)
  indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

  x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
  values = tf.reshape(x1, [-1])

  delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
  return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


# def unpool_argmax(pool, ind, ksize=(1, 2, 2, 1), scope='unpool'):
#     """
#        Unpooling layer after max_pool_with_argmax.
#        Args:
#            pool:   max pooled output tensor
#            ind:      argmax indices (produced by tf.nn.max_pool_with_argmax)
#            ksize:     ksize is the same as for the pool
#        Return:
#            unpooled:    unpooling tensor
#     """
#     with tf.variable_scope(scope):
#         pooled_shape = pool.get_shape().as_list()
#
#         flatten_ind = tf.reshape(ind, (pooled_shape[0], pooled_shape[1] * pooled_shape[2] * pooled_shape[3]))
#         # sparse indices to dense ones_like matrics
#         one_hot_ind = tf.one_hot(flatten_ind,  pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3], on_value=1., off_value=0., axis=-1)
#         one_hot_ind = tf.reduce_sum(one_hot_ind, axis=1)
#         one_like_mask = tf.reshape(one_hot_ind, (pooled_shape[0], pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2], pooled_shape[3]))
#         # resize input array to the output size by nearest neighbor
#         img = tf.image.resize_nearest_neighbor(pool, [pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2]])
#         unpooled = tf.multiply(img, tf.cast(one_like_mask, img.dtype))
#         return unpooled

def unpool_argmax(pool, ind, scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices (produced by tf.nn.max_pool_with_argmax)
       Return:
           unpooled:    unpooling tensor
    """
    with tf.variable_scope(scope):
        pooled_shape = pool.get_shape().as_list()
        assert pooled_shape[0] == 1
        flat_length = pooled_shape[1] * pooled_shape[2] * pooled_shape[3]

        flatten_ind = tf.reshape(ind, [flat_length, 1])
        flatten_pool = tf.reshape(pool, [flat_length])

        # sparse indices to dense ones_like matrics
        new_shape = [pooled_shape[0] * pooled_shape[1] * 2 * pooled_shape[2] * 2 * pooled_shape[3]]
        unpooled_vec = tf.scatter_nd(flatten_ind, flatten_pool, new_shape)
        unpooled = tf.reshape(unpooled_vec, [pooled_shape[0], pooled_shape[1] * 2, pooled_shape[2] * 2, pooled_shape[3]])
        return unpooled





