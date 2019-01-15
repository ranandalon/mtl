import tensorflow as tf
import utils.utils as utils
try:
  import tfplot
  tfplot_flag = True
except:
  print("Importing tfplot failed, will not plot instanceID outputs")
  tfplot_flag=False
FLAGS = tf.app.flags.FLAGS

def define_summaries(logits, ground_truths, processed_ground_truths, loss_op, loss_list, multi_loss_class):
  tf.summary.image('labelsID',   tf.cast(utils.rgb(processed_ground_truths[0]), tf.uint8))
  tf.summary.image('labelsID_org', tf.cast(ground_truths[0]*7, tf.uint8))   # *7 - histogram stretching
  tf.summary.image('InstanceID', ground_truths[1])
  tf.summary.image('Disparity',  tf.expand_dims(ground_truths[2][:, :, :, 0], axis=-1))
  if FLAGS.use_multi_loss:
    tf.summary.scalar('multi_class_loss', loss_op)
    for i, sigma_sq in zip(range(len(multi_loss_class._sigmas_sq)), multi_loss_class._sigmas_sq):
      tf.summary.scalar(sigma_sq.name, sigma_sq)
      tf.summary.scalar('wight' + str(i), 1 / (2 * sigma_sq))
  else:
    tf.summary.scalar('total_loss', loss_op)
  loss_names = []
  if FLAGS.use_label_type:
    loss_names.append('labelIds_loss')
    labelId_image_gray = utils.labelId(logits[0])
    tf.summary.image('output_labelIds_gray_scale', tf.cast(labelId_image_gray*7, tf.uint8))
    labelId_image = utils.rgb(logits[0], need_resize=FLAGS.need_resize)
    tf.summary.image('output_labelIds_color', tf.cast(labelId_image, tf.uint8), max_outputs=3)
    accuracy_lId_op = utils.accuracy(logits[0], processed_ground_truths[0])
    tf.summary.scalar('accuracy_labelId', accuracy_lId_op)
  if FLAGS.use_label_inst:
    if tfplot_flag:
      mask = tf.slice(ground_truths[1], [0, 0, 0, 2], [FLAGS.batch, FLAGS.output_height, FLAGS.output_width, 1])
      if FLAGS.need_resize:
        mask = tf.image.resize_images(mask, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      mask = tf.squeeze(mask, axis=3)
      summary_heatmap = tfplot.summary.wrap(calc_heat_map, batch=True)
      loss_names.append('InstanceID_loss')
      xy_input = calc_InstanceID_image(ground_truths[1], is_training=True, need_resize=FLAGS.need_resize)
      xy_output = calc_InstanceID_image(logits[1], is_training=True, need_resize=FLAGS.need_resize)
      summary_heatmap('input_InstanceID_x', xy_input[0])
      summary_heatmap('input_InstanceID_y', xy_input[1])
      summary_heatmap('output_InstanceID_x', tf.multiply(mask, xy_output[0]))
      summary_heatmap('output_InstanceID_y', tf.multiply(mask, xy_output[1]))
  if FLAGS.use_label_disp:
    loss_names.append('Disparity_loss')
    tf.summary.image('output_Disparity', tf.slice(logits[2], [0, 0, 0, 0], [FLAGS.batch, FLAGS.output_height, FLAGS.output_width, 1]))
  for loss, loss_name in zip(loss_list, loss_names):
    tf.summary.scalar(loss_name, loss)


def handle_summarys(sess, logits, summary, summary_writer, epoch, feed_dict):
  if epoch % FLAGS.summary_epoch == 0:
    summary_str = sess.run(summary, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, epoch)
    summary_writer.flush()


def calc_heat_map(data, cmap='jet'):
  fig, ax = tfplot.subplots()
  ax.imshow(data, cmap=cmap)
  return fig


def calc_InstanceID_image(raw_instance_id, is_training=True, need_resize=False):
  y_image = tf.squeeze(tf.slice(raw_instance_id, [0, 0, 0, 0], [FLAGS.batch, FLAGS.output_height, FLAGS.output_width, 1]), axis=3)
  x_image = tf.squeeze(tf.slice(raw_instance_id, [0, 0, 0, 1], [FLAGS.batch, FLAGS.output_height, FLAGS.output_width, 1]), axis=3)
  if is_training:
    InstanceID_image = 0
  else:
    InstanceID_image = 0
  return [x_image, y_image]