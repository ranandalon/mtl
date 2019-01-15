from inputs import inputs, get_records_num
from tqdm import tqdm

import classifier
import config
import tensorflow as tf
from utils import utils
import numpy as np
import os
from scipy.misc import imsave, imresize

# test_file, test_labels_file = utils.get_test_set(config.working_dataset, include_labels=True)
test_file, test_labels_file = utils.get_test_set(config.working_dataset, include_labels=True)

tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('test', test_file, 'Validation data')
tf.app.flags.DEFINE_string('test_labels', test_labels_file, 'Validation labels data')
tf.app.flags.DEFINE_string('test_logs', './logs/test', 'Validation Log directory')
tf.app.flags.DEFINE_integer('batch', 1, 'Batch size')
tf.app.flags.DEFINE_string('test_results_path', './labelIDResults', 'Save path to test images classification results')

FLAGS = tf.app.flags.FLAGS

def test():
  images, images_src = inputs(FLAGS.batch, FLAGS.test)
  c = get_records_num(FLAGS.test)
  # tf.summary.image('labels', labels[:, 1])
  # one_hot_labels = classifier.one_hot(labels[:, 1], is_color=False)

  autoencoder = utils.get_autoencoder(config.autoencoder, config.working_dataset, config.strided)
  logits = autoencoder.inference(images)

  labelImage_pred = classifier.labelId(logits)

  # accuracy_op = accuracy(logits, one_hot_labels)
  # tf.summary.scalar('accuracy', accuracy_op)

  saver = tf.train.Saver(tf.global_variables())
  # summary = tf.summary.merge_all()
  # summary_writer = tf.summary.FileWriter(FLAGS.val_logs)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
  session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  with tf.Session(config=session_config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not (ckpt and ckpt.model_checkpoint_path):
      print('No checkpoint file found')
      return

    ckpt_path = ckpt.model_checkpoint_path
    saver.restore(sess, ckpt_path)

    if not os.path.exists(FLAGS.test_results_path):
      os.mkdir(FLAGS.test_results_path)
    print('Working on ' + str(c) + ' images')
    for step in tqdm(range(np.ceil(c / FLAGS.batch).astype(int))):
      labelImages, filenames = sess.run([labelImage_pred, images_src])
      filenames = [x.decode('utf-8') for x in filenames]
      [imsave(os.path.join(FLAGS.test_results_path, fn), imresize(np.squeeze(x), [1024, 2048]))
      for x, fn in zip(labelImages, filenames)]

    print("Finished Creating Test Set labelID images")
    coord.request_stop()
    coord.join(threads)

def main(argv=None):
  utils.restore_logs(FLAGS.test_logs)
  test()

if __name__ == '__main__':
  tf.app.run()