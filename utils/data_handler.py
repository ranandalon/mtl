import tensorflow as tf
import numpy as np
import scipy.io
import user_config
import os
FLAGS = tf.app.flags.FLAGS


def init_data(num_of_imgs):
  input_images = [None]*num_of_imgs
  gt = [None]*num_of_imgs
  return input_images, gt

def get_data(ind, set_name):
  if set_name == 'val':
    flags_dict = {'input': FLAGS.val_input_file, 'labels': FLAGS.val_labels_type_files,
                  'labels_instance': FLAGS.val_labels_instance_files,
                  'disparity': FLAGS.val_labels_disparity_files,
                  'disparity_mask': FLAGS.val_disparity_mask_files}
  elif set_name == 'train':
    flags_dict = {'input': FLAGS.train_input_file, 'labels': FLAGS.train_labels_type_files,
                  'labels_instance': FLAGS.train_labels_instance_files,
                  'disparity': FLAGS.train_labels_disparity_files,
                  'disparity_mask': FLAGS.train_disparity_mask_files}
  elif set_name == 'example':
    flags_dict = {'input': FLAGS.example_input_file, 'labels': FLAGS.example_labels_type_files,
                  'labels_instance': FLAGS.example_labels_instance_files,
                  'disparity': FLAGS.example_labels_disparity_files,
                  'disparity_mask': FLAGS.example_disparity_mask_files}
  gt = [None]*3
  input_image = np.expand_dims(np.float32(scipy.misc.imread(os.path.join(flags_dict['input'], "%08d.png" % ind))), axis=0)
  gt[0] = np.expand_dims(np.uint8(scipy.misc.imread(os.path.join(flags_dict['labels'], "%08d.png" % ind))), axis=0)
  gt[0] = np.expand_dims(gt[0], axis=-1)
  #gt[1] = np.expand_dims(np.float32(scipy.misc.imread(FLAGS.train_train_labels_instance_files + "/%08d.tiff" % ind)), axis=0)
  yx_mask = np.load(os.path.join(flags_dict['labels_instance'], "%08d.npy" % ind))
  if FLAGS.need_resize:
    yx_mask[0] = yx_mask[0] / (FLAGS.input_height / FLAGS.output_height)
    yx_mask[1] = yx_mask[1] / (FLAGS.input_width / FLAGS.output_width)
  gt[1] = np.expand_dims(yx_mask, axis=0)
  disp_img = np.expand_dims(np.float32(scipy.misc.imread(os.path.join(flags_dict['disparity'], "%08d.png" % ind))), axis=0)
  disp_img = disp_img/FLAGS.max_disp_pix  # 32257
  disp_mask = np.expand_dims(np.float32(scipy.misc.imread(os.path.join(flags_dict['disparity_mask'], "%08d.png" % ind))), axis=0)
  gt[2] = np.concatenate([np.expand_dims(disp_img, axis=-1), np.expand_dims(disp_mask, axis=-1)], axis=-1)
  return input_image, gt

def get_all_data(set_name):
  num_of_imgs = calc_num_of_images(set_name)
  input_img, gt = init_data(num_of_imgs)
  for ind in range(num_of_imgs):
      if input_img[ind] is None:
          input_img[ind], gt[ind] = get_data(ind, set_name)
  return input_img, gt

def get_place_holders():
  #with tf.variable_scope(tf.get_variable_scope()):
  batch = None
  height = None
  width = None
  if user_config.autoencoder == 'SegNet':
    batch = 1
    height = FLAGS.input_height
    width = FLAGS.input_width
  input_ph = tf.placeholder(tf.float32, [batch, height, width, 3], name='input_ph')
  if FLAGS.input_pre_process == 'resNet_pre':
    pre_processed_input = tf.multiply(tf.subtract(tf.divide(input_ph, 255), 0.5), 2)
  else:
    pre_processed_input = input_ph
  labelsID_ph = tf.placeholder(tf.float32, [batch, None, None, 1], name='labelsID_ph')  # TODO: add 3 to config
  InstanceID_ph = tf.placeholder(tf.float32, [batch, None, None, 3], name='InstanceID_ph')  # TODO: add 3 to config
  Disparity_ph = tf.placeholder(tf.float32, [batch, None, None, 2], name='Disparity_ph')  # TODO: add 1 to config
  if FLAGS.need_resize:
    labelsID_out = tf.image.resize_images(labelsID_ph, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
    InstanceID_out = tf.image.resize_images(InstanceID_ph, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    Disparity_out = tf.image.resize_images(Disparity_ph, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  else:
    labelsID_out = labelsID_ph
    InstanceID_out = InstanceID_ph
    Disparity_out = Disparity_ph
  ground_truths_ph = [labelsID_ph, InstanceID_ph, Disparity_ph]
  ground_truths    = [labelsID_out, InstanceID_out, Disparity_out]
  return input_ph, ground_truths_ph, ground_truths, pre_processed_input


def calc_num_of_images(set_name):
  if set_name == 'val':
    return FLAGS.num_of_val_imgs
  if set_name == 'train':
    return FLAGS.num_of_train_imgs
  if set_name == 'example':
    return FLAGS.num_of_example_imgs