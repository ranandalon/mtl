from tqdm import tqdm
import config
import user_config
import tensorflow as tf
from utils import utils, loss_handler as lh, data_handler as dh, summary_handler as sh
import os
import numpy as np
from src.statistic import Statistic
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

train_output_files = [FLAGS.train_labels_type_files,
                      FLAGS.train_labels_instance_files,
                      FLAGS.train_labels_disparity_files]

results_dir = FLAGS.results_dir

def train():
  # create place holder img
  input_ph, ground_truths_ph, ground_truths, pre_processed_input = dh.get_place_holders()
  # Processing LabelId's
  one_hot_labels = utils.one_hot(ground_truths[0], is_color=False)  # TODO: add dictionary task-to-label-number
  # Geting model
  autoencoder = utils.get_autoencoder(user_config.autoencoder, config.working_dataset, config.strided)

  logits = autoencoder.inference(pre_processed_input)

  processed_ground_truths = [one_hot_labels, ground_truths[1], ground_truths[2]]
  loss_op, loss_list, multi_loss_class = lh.get_loss(logits, processed_ground_truths)

  optimizer = tf.train.AdamOptimizer(FLAGS.leaning_rate)
  train_step = optimizer.minimize(loss_op)

  saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
  session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  if FLAGS.use_summary: sh.define_summaries(logits, ground_truths, processed_ground_truths, loss_op, loss_list, multi_loss_class)
  num_of_train_examples = FLAGS.num_of_train_imgs
  statistic = Statistic(logits, loss_op, loss_list, input_ph, ground_truths_ph, multi_loss_class, processed_ground_truths)
  val_input_img, val_gt = dh.init_data(FLAGS.num_of_val_imgs)

  for ind in range(FLAGS.num_of_val_imgs):
    val_input_img[ind], val_gt[ind] = dh.get_data(ind, 'val')
  with tf.Session(config=session_config) as sess:
    global_step = start_training(sess, autoencoder, saver)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(results_dir + '/logs', sess.graph)
    input_img, gt = dh.init_data(num_of_train_examples)
    input_batch = None
    # training starts here
    step = 0
    for epoch in range(FLAGS.num_of_epchs):
      print("\nEpoch: " + str(epoch))
      sub_batche = 0
      for ind in tqdm(np.random.permutation(num_of_train_examples)):
        if input_img[ind] is None:
          input_img[ind], gt[ind] = dh.get_data(ind, 'train')
        # ----- make the random batch ----
        if sub_batche == 0:
          input_batch = input_img[ind]
          gt_batch = gt[ind]
        else:
          input_batch, gt_batch = add_to_batch(input_batch, gt_batch, input_img[ind], gt[ind])
        if sub_batche < FLAGS.batch-1:
          sub_batche +=1
          continue
        sub_batche = 0
        # ---- batch is ready ----
        feed_dict = get_feed_dict(input_ph, ground_truths_ph, input_batch, gt_batch)
        sess.run(train_step, feed_dict=feed_dict)
        if FLAGS.use_summary and step % FLAGS.calc_summary == 0:
          sh.handle_summarys(sess, logits, summary, summary_writer, step, feed_dict)
        step += 1
      statistic.handle_statistic(epoch, logits, sess, input_img, gt, val_input_img, val_gt)
      if epoch % FLAGS.epoch_model_ckpts == 0:
        ckpt_dir = os.path.join(results_dir, 'global_ckpt')
        if not os.path.exists(ckpt_dir):
          os.mkdir(ckpt_dir)
        saver.save(sess, os.path.join(ckpt_dir, 'global_ckpt'), global_step=global_step)
      if epoch % FLAGS.epoch_analysis_breakpoints == 0:
        analysis_ckpt_dir = os.path.join(results_dir, 'Analysis_ckpts')
        if not os.path.exists(analysis_ckpt_dir):
          os.mkdir(analysis_ckpt_dir)
        saver.save(sess, os.path.join(analysis_ckpt_dir, 'epoch_' + str(epoch)), global_step=global_step)
    # training ends here


def add_to_batch(input_batch, gt_batch, input_img, gt):
  input_batch_plus = np.concatenate([input_batch, input_img], axis=0)
  gt_batch_plus = []
  for ind in range(len(gt)):
    gt_batch_plus.append(np.concatenate([gt_batch[ind], gt[ind]], axis=0))
  return input_batch_plus, gt_batch_plus


def get_feed_dict(input_ph, ground_truths_ph, input, outputs):
  feed_dict = {input_ph: input}
  feed_dict[ground_truths_ph[0]] = outputs[0]
  feed_dict[ground_truths_ph[1]] = outputs[1]
  feed_dict[ground_truths_ph[2]] = outputs[2]
  return feed_dict


def start_training(sess, autoencoder, saver):
  ckpt = tf.train.get_checkpoint_state(os.path.join(results_dir, 'ckpts'))
  if user_config.autoencoder != 'ResNet':
    if not ckpt:
      print('No checkpoint file found. Initializing...')
      global_step = 0
      # autoencoder.restore(sess)  # TODO: need to add restore method to autoencoder
      init = tf.global_variables_initializer()
      sess.run(init)
    else:
      global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
      ckpt_path = ckpt.model_checkpoint_path
      saver.restore(sess, ckpt_path)
  else:
    global_step = 0
    autoencoder.restore(sess, restore_first=FLAGS.restart_training)
  return global_step


def main(argv=None):
  utils.restore_logs(os.path.join(results_dir, 'ckpts'))
  train()

if __name__ == '__main__':
  tf.app.run()
