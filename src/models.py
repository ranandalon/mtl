import utils
import src.convnet as cnn
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1, resnet_utils
import os
from tensorflow.python.ops import nn
import sys
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS


class NeuralNet:
  def conv(self, x, filter_size, channels_shape, name):
    return cnn.conv(x, filter_size, channels_shape, 1, name)

  def strided_conv(self, x, filter_size, channels_shape, stride, name):    #stride convolution
    return cnn.conv(x, filter_size, channels_shape, stride, name)

  def deconv(self, x, filter_size, channels_shape, name):
    return cnn.deconv(x, filter_size, channels_shape, 1, name)

  def deconvNoRelu(self, x, filter_size, channels_shape, name):
    return cnn.deconvNoRelu(x, filter_size, channels_shape, 1, name)

  def pool(self, x, size):
    return cnn.max_pool(x, size, size)

  def pool_argmax(self, x, size):
    return cnn.max_pool_argmax(x, size, size)

  def unpool(self, x, size):
    return cnn.unpool(x, size)

  def unpool_argmax(self, x, argmax):
    return cnn.unpool_argmax(x, argmax)

  def resize_conv(self, x, channels_shape, name):
    shape = x.get_shape().as_list()
    height = shape[1] * 2
    width = shape[2] * 2
    resized = tf.image.resize_nearest_neighbor(x, [height, width])
    return cnn.conv(resized, [3, 3], channels_shape, 1, name, repad=True)

class encoder(NeuralNet):
    def __init__(self, isArgmaxPooling=True):
      if isArgmaxPooling:
        self._argmax_list = []

    def encode_with_pooling(self, images):
        with tf.variable_scope('pool1'):
          conv1 = self.conv(images, [3, 3], [3, 64], 'conv1_1')
          conv2 = self.conv(conv1, [3, 3], [64, 64], 'conv1_2')
          pool1, argmax1 = self.pool_argmax(conv2, 2)
          self._argmax_list.append(argmax1)

        with tf.variable_scope('pool2'):
          conv3 = self.conv(pool1, [3, 3], [64, 128], 'conv2_1')
          conv4 = self.conv(conv3, [3, 3], [128, 128], 'conv2_2')
          pool2, argmax2 = self.pool_argmax(conv4, 2)
          self._argmax_list.append(argmax2)

        with tf.variable_scope('pool3'):
          conv5 = self.conv(pool2, [3, 3], [128, 256], 'conv3_1')
          conv6 = self.conv(conv5, [3, 3], [256, 256], 'conv3_2')
          conv7 = self.conv(conv6, [3, 3], [256, 256], 'conv3_3')
          pool3, argmax3 = self.pool_argmax(conv7, 2)
          self._argmax_list.append(argmax3)

        with tf.variable_scope('pool4'):
          conv8 = self.conv(pool3, [3, 3], [256, 512], 'conv4_1')
          conv9 = self.conv(conv8, [3, 3], [512, 512], 'conv4_2')
          conv10 = self.conv(conv9, [3, 3], [512, 512], 'conv4_3')
          pool4, argmax4 = self.pool_argmax(conv10, 2)
          self._argmax_list.append(argmax4)

        with tf.variable_scope('pool5'):
          conv11 = self.conv(pool4, [3, 3], [512, 512], 'conv5_1')
          conv12 = self.conv(conv11, [3, 3], [512, 512], 'conv5_2')
          conv13 = self.conv(conv12, [3, 3], [512, 512], 'conv5_3')
          pool5, argmax5 = self.pool_argmax(conv13, 2)
          self._argmax_list.append(argmax5)
        return pool5, self._argmax_list

    def encode_with_resnet(self, images, global_pool=False, output_stride=8):
        self.global_pool = global_pool  # needed for artus convolution
        self.output_stride = output_stride  # needed for artus convolution
        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_101(images,
                                                         global_pool=self.global_pool,
                                                         output_stride=self.output_stride)
            #size = tf.slice(tf.shape(images), [1], [2])  #TODO: multiply by 0.5
            size = [FLAGS.output_height, FLAGS.output_width]  #TODO: chenged fixed size
            resized_logits = tf.image.resize_images(logits, size, method=tf.image.ResizeMethod.BILINEAR,
                                                    align_corners=False)
        return resized_logits

class decoder(NeuralNet):
  def __init__(self, n, isArgmaxPooling=True, argmax_list=[]):
    '''

    :param n: output number
    :param isArgmaxPooling: if we want to upsample with argmax position reconstruction
    :param argmax_list: tensor of tensors for the argmax positions for each upsample block
    '''
    self._n = n
    if isArgmaxPooling:
      self._argmax_list = argmax_list
      assert len(self._argmax_list) == 5

  def decode_with_pooling(self, encoded_image):
      with tf.variable_scope('unpool1'):
        unpool1 = self.unpool_argmax(encoded_image, self._argmax_list[-1])
        deconv1 = self.deconv(unpool1, [3, 3], [512, 512], 'deconv5_3')
        deconv2 = self.deconv(deconv1, [3, 3], [512, 512], 'deconv5_2')
        deconv3 = self.deconv(deconv2, [3, 3], [512, 512], 'deconv5_1')

      with tf.variable_scope('unpool2'):
        unpool2 = self.unpool_argmax(deconv3, self._argmax_list[-2])
        deconv4 = self.deconv(unpool2, [3, 3], [512, 512], 'deconv4_3')
        deconv5 = self.deconv(deconv4, [3, 3], [512, 512], 'deconv4_2')
        deconv6 = self.deconv(deconv5, [3, 3], [256, 512], 'deconv4_1')

      with tf.variable_scope('unpool3'):
        unpool3 = self.unpool_argmax(deconv6, self._argmax_list[-3])
        deconv7 = self.deconv(unpool3, [3, 3], [256, 256], 'deconv3_3')
        deconv8 = self.deconv(deconv7, [3, 3], [256, 256], 'deconv3_2')
        deconv9 = self.deconv(deconv8, [3, 3], [128, 256], 'deconv3_1')

      with tf.variable_scope('unpool4'):
        unpool4 = self.unpool_argmax(deconv9, self._argmax_list[-4])
        deconv10 = self.deconv(unpool4, [3, 3], [128, 128], 'deconv2_2')
        deconv11 = self.deconv(deconv10, [3, 3], [64, 128], 'deconv2_1')

      with tf.variable_scope('unpool5'):
        unpool5 = self.unpool_argmax(deconv11, self._argmax_list[-5])
        deconv12 = self.deconv(unpool5, [3, 3], [64, 64], 'deconv1_2')
        deconv13 = self.deconv(deconv12, [3, 3], [self._n, 64], 'deconv1_1')
        deconv13_noRelu = self.deconvNoRelu(deconv12, [3, 3], [self._n, 64], 'deconv1_1_NoRelu')
      return deconv13, deconv13_noRelu

  def decode_resnet(self, encoded_image, decode_num=0, num_of_output_channels=33, activation=True):
      with tf.variable_scope('decoder%d'%(decode_num)):
          padding = 'SAME'
          initializer = tf.truncated_normal_initializer(stddev=0.01)
          regularizer = slim.l2_regularizer(0.0005)
          conv0 = slim.conv2d(encoded_image, 512, [3, 3],  # 1024 channels
                            padding=padding,
                            weights_initializer=initializer,
                            weights_regularizer=regularizer,
                            scope='conv0')
          conv1 = slim.conv2d(conv0, 512, [1, 1],  # 256 channels
                            padding=padding,
                            weights_initializer=initializer,
                            weights_regularizer=regularizer,
                            scope='conv1')
          if activation:  #TODO fix: realy ugly patch (and False)
              conv2 = slim.conv2d(conv1, num_of_output_channels, [1, 1],  # 33 channels (for 33 segmantations)
                                  padding=padding,
                                  weights_initializer=initializer,
                                  weights_regularizer=regularizer,
                                  scope='conv2')
          else:
              conv2 = slim.conv2d(conv1, num_of_output_channels, [1, 1],  # 33 channels (for 33 segmantations)
                                  padding=padding,
                                  weights_initializer=initializer,
                                  weights_regularizer=regularizer,
                                  activation_fn=None,
                                  scope='conv2')
      return conv2


class SegNetTest():
  def __init__(self, n_list, strided=False, decoders_num=3):
    self._decoders_num=decoders_num
    self._encoder = encoder()
    self._n = n_list
    self._logits = []
    self._loss = 0

  def inference(self, images):
    return self.inference_with_pooling(images)

  def inference_with_pooling(self, images):
    tf.summary.image('input', images, max_outputs=3)
    with tf.variable_scope('encoder'):
      encoded_images, argmax_lists = self._encoder.encode_with_pooling(images)

    for i in range(self._decoders_num):
      with tf.variable_scope('decoder_' + str(i)):
        self._decoder = decoder(self._n[i], isArgmaxPooling=True, argmax_list=argmax_lists)
        if i == 1:
          # For InstanceID we dont want ReLU as last activation layer (regressing negative distances as well)
          self._logits.append(self._decoder.decode_with_pooling(encoded_images)[1])
        else:
          self._logits.append(self._decoder.decode_with_pooling(encoded_images)[0])

    return self._logits

  def restore(self, sess, restore_first=True):
      if restore_first:
          sess.run(tf.global_variables_initializer())
          self.init_fn(sess)
      else:
          self.cont_fn(sess)


class Autoencoder:
  def __init__(self, n, strided=False, max_images=3):
    self.max_images = max_images
    self.n = n
    self.strided = strided

  def conv(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 1, name)

  def conv2(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 2, name)

  def deconv(self, x, channels_shape, name):
    return cnn.deconv(x, [3, 3], channels_shape, 1, name)

  def pool(self, x):
    return cnn.max_pool(x, 2, 2)

  def pool_argmax(self, x):
    return cnn.max_pool_argmax(x, 2, 2)

  def unpool(self, x):
    return cnn.unpool(x, 2)

  def unpool_argmax(self, x, argmax):
    return cnn.unpool_layer2x2_batch(x, argmax)

  def resize_conv(self, x, channels_shape, name):
    shape = x.get_shape().as_list()
    height = shape[1] * 2
    width = shape[2] * 2
    resized = tf.image.resize_nearest_neighbor(x, [height, width])
    return cnn.conv(resized, [3, 3], channels_shape, 1, name, repad=True)

  def inference(self, images):
    if self.strided:
      return self.strided_inference(images)
    return self.inference_with_pooling(images)

class MiniAutoencoder(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('encode2'):
      conv3 = self.conv(conv2, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('encode3'):
      conv5 = self.conv(conv4, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('decode1'):
      deconv7 = self.resize_conv(conv7, [256, 256], 'deconv3_3')
      deconv6 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv5 = self.deconv(deconv6, [128, 256], 'deconv3_1')

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(deconv5, [128, 128], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [64, 128], 'deconv2_1')

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(deconv3, [64, 64], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 64], 'deconv1_1')

    return deconv1

class SegNetAutoencoder(Autoencoder):
  def __init__(self, n, strided=False, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def inference_with_pooling(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv(conv1, [64, 64], 'conv1_2')
      pool1 = self.pool(conv2)

    with tf.variable_scope('pool2'):
      conv3 = self.conv(pool1, [64, 128], 'conv2_1')
      conv4 = self.conv(conv3, [128, 128], 'conv2_2')
      pool2 = self.pool(conv4)

    with tf.variable_scope('pool3'):
      conv5 = self.conv(pool2, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv(conv6, [256, 256], 'conv3_3')
      pool3 = self.pool(conv7)

    with tf.variable_scope('pool4'):
      conv8 = self.conv(pool3, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv(conv9, [512, 512], 'conv4_3')
      pool4 = self.pool(conv10)

    with tf.variable_scope('pool5'):
      conv11 = self.conv(pool4, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv(conv12, [512, 512], 'conv5_3')
      pool5 = self.pool(conv13)

    with tf.variable_scope('unpool1'):
      unpool1 = self.unpool(pool5)
      deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      unpool2 = self.unpool(deconv3)
      deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      unpool3 = self.unpool(deconv6)
      deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      unpool4 = self.unpool(deconv9)
      deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      unpool5 = self.unpool(deconv11)
      deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    return deconv13

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('pool2'):
      conv3 = self.conv(conv2, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('pool3'):
      conv5 = self.conv(conv4, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('pool4'):
      conv8 = self.conv(conv7, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv2(conv9, [512, 512], 'conv4_3')

    with tf.variable_scope('pool5'):
      conv11 = self.conv(conv10, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv2(conv12, [512, 512], 'conv5_3')

    with tf.variable_scope('unpool1'):
      deconv1 = self.resize_conv(conv13, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      deconv4 = self.resize_conv(deconv3, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      deconv7 = self.resize_conv(deconv6, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      deconv10 = self.resize_conv(deconv9, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      deconv12 = self.resize_conv(deconv11, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    return deconv13

class SegNetArgmaxAE(Autoencoder):
  def __init__(self, n, strided=False, max_images=3, decoders_num=1, decoders_type=['c']):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)
  def inference_with_pooling(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv(conv1, [64, 64], 'conv1_2')
      pool1, argmax1 = self.pool_argmax(conv2)

    with tf.variable_scope('pool2'):
      conv3 = self.conv(pool1, [64, 128], 'conv2_1')
      conv4 = self.conv(conv3, [128, 128], 'conv2_2')
      pool2, argmax2 = self.pool_argmax(conv4)

    with tf.variable_scope('pool3'):
      conv5 = self.conv(pool2, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv(conv6, [256, 256], 'conv3_3')
      pool3, argmax3 = self.pool_argmax(conv7)

    with tf.variable_scope('pool4'):
      conv8 = self.conv(pool3, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv(conv9, [512, 512], 'conv4_3')
      pool4, argmax4 = self.pool_argmax(conv10)

    with tf.variable_scope('pool5'):
      conv11 = self.conv(pool4, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv(conv12, [512, 512], 'conv5_3')
      pool5, argmax5 = self.pool_argmax(conv13)

    with tf.variable_scope('unpool1'):
      unpool1 = self.unpool_argmax(pool5, argmax5)
      deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      unpool2 = self.unpool_argmax(deconv3, argmax4)
      deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      unpool3 = self.unpool_argmax(deconv6, argmax3)
      deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      unpool4 = self.unpool_argmax(deconv9, argmax2)
      deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      unpool5 = self.unpool_argmax(deconv11, argmax1)
      deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    return deconv13

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('pool2'):
      conv3 = self.conv(conv2, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('pool3'):
      conv5 = self.conv(conv4, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('pool4'):
      conv8 = self.conv(conv7, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv2(conv9, [512, 512], 'conv4_3')

    with tf.variable_scope('pool5'):
      conv11 = self.conv(conv10, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv2(conv12, [512, 512], 'conv5_3')

    with tf.variable_scope('unpool1'):
      deconv1 = self.resize_conv(conv13, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      deconv4 = self.resize_conv(deconv3, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      deconv7 = self.resize_conv(deconv6, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      deconv10 = self.resize_conv(deconv9, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      deconv12 = self.resize_conv(deconv11, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    return deconv13


class ResNetAutoencoder():
    def __init__(self, n_list, strided=False, decoders_num=3):
        self._decoders_num = decoders_num
        self._encoder = encoder()
        self._decoder = decoder(n_list, isArgmaxPooling=False)
        self._n = n_list
        self._logits = []
        self._includes = []
        self._resnet_checkpoint_path = FLAGS.res_net_101_ckpt
        self._checkpoint_path = os.path.join(FLAGS.results_dir, 'global_ckpt')
        if not os.path.exists(self._resnet_checkpoint_path):
            raise NameError("Oops! No valid resnet_checkpoint_path")

    def inference(self, images):
        # TODO: pass as parameter the output channels num - according to GT
        tf.summary.image('input', tf.cast(tf.multiply(tf.add(tf.multiply(images, 0.5), 0.5), 255), tf.uint8),
                         max_outputs=3)
        encoded_images = self._encoder.encode_with_resnet(images)
        num_of_output_channels = [34, 2, 1]
        for i in range(self._decoders_num):
            if i in [0, 1, 2]:  #TODO: check 2
                self._logits.append(self._decoder.decode_resnet(encoded_images, decode_num=i,
                                                                num_of_output_channels=num_of_output_channels[i],
                                                                activation=False))
            else:
                self._logits.append(self._decoder.decode_resnet(encoded_images, decode_num=i,
                                                                num_of_output_channels=num_of_output_channels[i]))
            self._includes.append('decoder%d' % (i))

        self._weights_restored_from_file = slim.get_variables_to_restore()
        self._weights_randomly_initialized = slim.get_variables_to_restore(include=self._includes)
        self._all_weights = self._weights_restored_from_file + self._weights_randomly_initialized

        # function for first init
        self.init_fn = slim.assign_from_checkpoint_fn(self._resnet_checkpoint_path,
                                                      self._weights_restored_from_file,
                                                      self._weights_randomly_initialized)
        self.init_op = tf.variables_initializer(self._weights_randomly_initialized)

        # function for restoring traind model
        #self.cont_fn = slim.assign_from_checkpoint_fn(self._checkpoint_path,
        #                                              self._all_weights)
        return self._logits

    def restore(self, sess, restore_first=True, saver=None, checkpoint=None):
        if restore_first:
            sess.run(self.init_op)
            sess.run(tf.global_variables_initializer())
            self.init_fn(sess)
        else:
            if checkpoint != None:
                ckpt_path = checkpoint.model_checkpoint_path
            else:
                ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
                ckpt_path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt_path)
            # self.cont_fn(sess)







