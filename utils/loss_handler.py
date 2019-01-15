import tensorflow as tf
from utils.utils import accuracy
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

class MultiLossLayer():
  def __init__(self, loss_list):
    self._loss_list = loss_list
    self._sigmas_sq = []
    for i in range(len(self._loss_list)):
      self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

  def get_loss(self):
    factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))
    loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0]))
    for i in range(1, len(self._sigmas_sq)):
      factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
      loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i])))
    return loss

def get_loss(logits, ground_truths):
  multi_loss_class = None
  loss_list = []
  if FLAGS.use_label_type:
    if FLAGS.need_resize:
      label_type = tf.image.resize_images(ground_truths[0], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    else:
      label_type = ground_truths[0]
    loss_list.append(loss(logits[0], label_type, type='cross_entropy'))
  if FLAGS.use_label_inst:
    xy_gt = tf.slice(ground_truths[1], [0, 0, 0, 0], [-1, FLAGS.output_height, FLAGS.output_width, 2])    # to get x GT and y GT
    mask = tf.slice(ground_truths[1], [0, 0, 0, 2], [-1, FLAGS.output_height, FLAGS.output_width, 1])  # to get mask from GT
    mask = tf.concat([mask, mask], 3)  # to get mask for x and for y
    if FLAGS.need_resize:
      xy_gt = tf.image.resize_images(xy_gt, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      mask = tf.image.resize_images(mask, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    loss_list.append(l1_masked_loss(tf.multiply(logits[1], mask), xy_gt, mask))
  if FLAGS.use_label_disp:
    if FLAGS.need_resize:
      gt_sized = tf.image.resize_images(ground_truths[2], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      gt_sized = gt_sized[:, :, :, 0]
      mask = gt_sized[:, :, :, 1]
    else:
      gt_sized = tf.expand_dims(ground_truths[2][:, :, :, 0], axis=-1)
      mask = tf.expand_dims(ground_truths[2][:, :, :, 1], axis=-1)
    loss_list.append(l1_masked_loss(tf.multiply(logits[2], mask), tf.multiply(gt_sized, mask), mask))
  if FLAGS.use_multi_loss:
    loss_op, multi_loss_class = calc_multi_loss(loss_list)
  else:
    loss_op = loss_list[0]
    for i in range(1, len(loss_list)):
      loss_op = tf.add(loss_op, loss_list[i])
  return loss_op, loss_list, multi_loss_class

def calc_multi_loss(loss_list):
  multi_loss_layer = MultiLossLayer(loss_list)
  return multi_loss_layer.get_loss(), multi_loss_layer

def l1_masked_loss(logits, gt, mask):
  valus_diff = tf.abs(tf.subtract(logits, gt))
  L1_loss = tf.divide(tf.reduce_sum(valus_diff), tf.add(tf.reduce_sum(mask[:, :, :, 0]), 0.0001))
  return L1_loss

def loss(logits, labels, type='cross_entropy'):
  if type == 'cross_entropy':
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy, name='loss')
  if type == 'l2':
    return tf.nn.l2_loss(tf.subtract(logits, labels))
  if type == 'l1':
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(logits, labels)), axis=-1))



