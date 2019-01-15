from src.models import MiniAutoencoder, SegNetAutoencoder, SegNetArgmaxAE, SegNetTest, ResNetAutoencoder
import config
import tensorflow as tf
from functools import reduce
import os
import utils.data_handler as dh
import numpy as np
import matplotlib.pyplot as plt
import src.OPTICS as OPTICS

colors = tf.cast(tf.stack(config.colors[config.working_dataset]), tf.float32)  # / 255
FLAGS = tf.app.flags.FLAGS

def get_autoencoder(autoencoder_name, dataset_name, strided):
  n_labels = len(config.colors[dataset_name])
  autoencoders = {
    'mini': MiniAutoencoder,
    'SegNet': SegNetTest,
    'CityScapes' : SegNetTest,
    'CityScapes_old' : SegNetAutoencoder,
    'ResNet': ResNetAutoencoder
  }
  if autoencoder_name == 'SegNet':
    n_labels = [n_labels, 2, 1]
  return autoencoders[autoencoder_name](n_labels, strided=strided)


def restore_logs(logfile):
  '''
  Fixed - will now not delete existing log files but add sub-index to path
  :param logfile:
  :return:
  '''
  if tf.gfile.Exists(logfile):
    print('logfile already exist: %s' % logfile)
    # i = 1
    # while os.path.exists(logfile + '_' + str(i)):
    #   i += 1
    # logfile = logfile + '_' + str(i)
    # print('Creating anf writing to: %s' % logfile)
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)


def color_mask(tensor, color):
  return tf.reduce_all(tf.equal(tensor, color), 3)

def one_hot(labels, is_color=True):
  if is_color:
    color_tensors = tf.unstack(colors)
    channel_tensors = list(map(lambda color: color_mask(tf.cast(labels, tf.float32), color), color_tensors))

    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
  else:
    # TODO: Need to create images of each label from 1 to 33 in size of label image
    colors_labelIds = tf.cast(tf.range(len(config.colors[config.working_dataset])), tf.float32)
    color_tensors = tf.unstack(colors_labelIds)
    channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
  return one_hot_labels

def rgb(logits, need_resize=False):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, -1)
  n = colors.get_shape().as_list()[0]
  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  one_hot_matrix = tf.reshape(one_hot, [-1, n])
  rgb_matrix = tf.matmul(one_hot_matrix, colors)
  rgb_tensor = tf.reshape(rgb_matrix, [-1, FLAGS.output_height, FLAGS.output_width, 3])
  return tf.cast(rgb_tensor, tf.float32)

def labelId(logits):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)
  argmax_expand = tf.expand_dims(argmax, -1)
  return tf.cast(argmax_expand*7, tf.float32)

def disparity(logits):
  return tf.cast(logits, tf.float32)

def onehot_to_rgb(one_hot):
  n = colors.get_shape().as_list()[0]
  one_hot_matrix = tf.reshape(one_hot, [-1, n])
  rgb_matrix = tf.matmul(one_hot_matrix, colors)
  rgb_tensor = tf.reshape(rgb_matrix, [-1, FLAGS.output_height, FLAGS.output_width, 3])
  return tf.cast(rgb_tensor, tf.float32)


def accuracy(logits, labels):
  if FLAGS.need_resize:
    labels = tf.image.resize_images(labels, [FLAGS.output_height, FLAGS.output_width],
                                    method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  softmax = tf.nn.softmax(logits, dim=3)
  argmax = tf.argmax(softmax, 3)

  shape = logits.get_shape().as_list()
  n = shape[3]

  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
  total_pixels = reduce(lambda x, y: x * y, [FLAGS.batch] + shape[1:3])
  return equal_pixels / total_pixels


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_model(autoencoder_name='ResNet'):
    input_ph, ground_truths_ph, ground_truths, pre_processed_input = dh.get_place_holders()
    autoencoder = get_autoencoder(autoencoder_name, config.working_dataset, config.strided)
    logits = autoencoder.inference(pre_processed_input)
    return autoencoder, logits, input_ph, ground_truths_ph, ground_truths, pre_processed_input


def get_run_list(logits, INF_FLAGS):
    run_list = []
    if INF_FLAGS['use_label_type']:
        labelId_image = rgb(logits[0])
        run_list.append(tf.cast(labelId_image, tf.uint8))
    if INF_FLAGS['use_label_inst']:
        run_list.append(logits[1])
    if INF_FLAGS['use_label_disp']:
        run_list.append(logits[2])
    return run_list

def pred_list2dict(pred_list, INF_FLAGS):
    pred_dict = {}
    if INF_FLAGS['use_label_disp']:
        image = np.expand_dims(pred_list.pop().squeeze().clip(max=1, min=0)*255, 2).astype('uint8')
        image = np.concatenate([image, image, image], axis=2)
        pred_dict['disp'] = image
    if INF_FLAGS['use_label_inst']:
        pred_dict['instance'] = pred_list.pop().squeeze()
    if INF_FLAGS['use_label_type']:
        pred_dict['label'] = pred_list.pop().squeeze()
    return pred_dict


def calc_instance(label_arr, xy_arr):
    mask = make_mask(label_arr)
    raw_image = np.concatenate([xy_arr, np.expand_dims(mask, axis=2)], axis=2)
    instance_image = OPTICS.calc_clusters_img(raw_image)
    return instance_image.clip(max=255, min=0).astype('uint8')


def make_mask(label_image):
    ids = [24, 26]
    for i, id in enumerate(ids):
        color = config.colors[config.working_dataset][id]
        mask = label_image == color
        mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
        if i == 0:
            total_mask = mask
        else:
            total_mask = total_mask + mask
    return total_mask

