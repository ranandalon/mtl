from datetime import datetime
import os
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train', '/home/alon-ran/CityScapes/leftImg8bit/train', 'Training images directory')
tf.app.flags.DEFINE_string('train_labels_type', '/home/alon-ran/CityScapes/gtFine/train', 'Training fine type label images directory')
tf.app.flags.DEFINE_string('train_labels_instance', '/home/alon-ran/CityScapes/gtFine/train', 'Training fine instance label images directory')
tf.app.flags.DEFINE_string('train_labels_coarse', '/home/alon-ran/CityScapes/gtCoarse/train', 'Training coarse label images directory')
tf.app.flags.DEFINE_string('train_labels_disparity', '/home/alon-ran/CityScapes/disparity/train', 'Training disparity label images directory')

tf.app.flags.DEFINE_string('test', '/home/alon-ran/CityScapes/leftImg8bit/test', 'Test images directory')
tf.app.flags.DEFINE_string('test_labels_type', '/home/alon-ran/CityScapes/gtFine/test', 'Test fine type label images directory')
tf.app.flags.DEFINE_string('test_labels_instance', '/home/alon-ran/CityScapes/gtFine/test', 'Test fine instance label images directory')
# tf.app.flags.DEFINE_string('test_labels_coarse', '/home/alon-ran/CityScapes/gtCoarse/test', 'Test coarse label images directory')
tf.app.flags.DEFINE_string('test_labels_disparity', '/home/alon-ran/CityScapes/disparity/test', 'Test disparity label images directory')

tf.app.flags.DEFINE_string('val', '/home/alon-ran/CityScapes/leftImg8bit/val', 'Validation images directory')
tf.app.flags.DEFINE_string('val_labels_type', '/home/alon-ran/CityScapes/gtFine/val', 'Validation fine type label images directory')
tf.app.flags.DEFINE_string('val_labels_instance', '/home/alon-ran/CityScapes/gtFine/val', 'Validation fine instance label images directory')
# tf.app.flags.DEFINE_string('test_labels_coarse', '/home/alon-ran/CityScapes/gtCoarse/test', 'Test coarse label images directory')
tf.app.flags.DEFINE_string('val_labels_disparity', '/home/alon-ran/CityScapes/disparity/val', 'Validation disparity label images directory')

tf.app.flags.DEFINE_string('labels_file', 'labels', 'Labels file')
tf.app.flags.DEFINE_string('output', 'input/CityScapes', 'Output data directory')
tf.app.flags.DEFINE_boolean('encode_test_images', True, 'Encode test images')
tf.app.flags.DEFINE_boolean('encode_val_images', True, 'Encode test images')
tf.app.flags.DEFINE_integer('train_shards', 1, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('test_shards', 1, 'Number of shards in test TFRecord files')
tf.app.flags.DEFINE_integer('val_shards', 1, 'Number of shards in validation TFRecord files')
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to preprocess the images')

FLAGS = tf.app.flags.FLAGS
IGNORE_FILENAMES = ['.DS_Store']


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': _bytes_feature(image_buffer),
    'image/src': _bytes_feature(tf.compat.as_bytes(filename.split('/')[-1]))
  }))

  return example


class ImageCoder(object):
  def __init__(self, gray, isInstance=False):
    self._sess = tf.Session()
    self.isGrayscale = gray
    self._png_data = tf.placeholder(dtype=tf.string)
    if isInstance:
      self._isInstance = True
      self._image = tf.placeholder(dtype=tf.uint8)
      self._decode_png = tf.image.decode_png(self._png_data, channels=0, dtype=tf.uint16)
      self._decode_png = tf.image.resize_images(tf.cast(self._decode_png, tf.float32), size=[256, 512])
      self._encode_png = tf.image.encode_png(self._image)
    elif self.isGrayscale:
      self._image = tf.placeholder(dtype=tf.uint8)
      self._decode_png = tf.image.decode_png(self._png_data, channels=0)
      self._decode_png = tf.image.resize_images(self._decode_png, size=[256, 512])
      self._encode_png = tf.image.encode_png(self._image)
    else:
      self._image = tf.placeholder(dtype=tf.uint8)
      self._decode_png = tf.image.decode_png(self._png_data, channels=3)
      self._decode_png = tf.image.resize_images(self._decode_png, size=[256, 512])
      self._encode_png = tf.image.encode_png(self._image)

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png, feed_dict={self._png_data: image_data})
    if self._isInstance:
      image = regress_centers(image)
      self.isGrayscale = False  # now 3 channel image - [dx, dy, mask]
    assert len(image.shape) == 3
    if not self.isGrayscale:
      assert image.shape[2] == 3
    return image

  def encode_png(self, image):
    image_data = self._sess.run(self._encode_png, feed_dict={self._image: image})
    return image_data


def regress_centers(Image):
  Image = np.squeeze(Image)
  instances = np.unique(Image)
  instances = instances[instances > 1000]

  mask = np.zeros_like(Image)
  mask[np.where(Image > 1000)] = 1

  centroid_regression = np.zeros([Image.shape[0], Image.shape[1], 3])
  centroid_regression[:, :, 2] = mask

  for instance in instances:
    # step A - get a center (x,y) for each instance
    instance_pixels = np.where(Image == instance)
    y_c, x_c = np.mean(instance_pixels[0]), np.mean(instance_pixels[1])
    # step B - calculate dist_x, dist_y of each pixel of instance from its center
    y_dist = (instance_pixels[0] - y_c + Image.shape[0])
    x_dist = (instance_pixels[1] - x_c + Image.shape[1])
    for y, x, d_y, d_x in zip(instance_pixels[0], instance_pixels[1], y_dist, x_dist):
      centroid_regression[y, x, :2] = [d_y, d_x]  # remember - y is distance in rows, x in columns
  return centroid_regression


def _process_image(filename, coder):
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  image = coder.decode_png(image_data)
  image_downscaled = coder.encode_png(image)
  assert len(image.shape) == 3
  if not coder.isGrayscale:
    assert image.shape[2] == 3
  height, width, _ = image.shape

  return image_downscaled, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards):
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    if num_shards == 1:
      output_filename = '%s.tfrecords' % name
    else:
      output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output, output_filename)
    if tf.gfile.Exists(output_file):
      continue
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]

      if filename.split('/')[-1] in IGNORE_FILENAMES:
        continue

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards, gray, isInstance):
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder(gray=gray, isInstance=isInstance)

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _process_dataset(name, directory, num_shards, gray=True):
  if 'type' in name:
    file_path = '%s/*/*labelIds.png' % directory
    isInstance=False
  elif 'instance' in name:
    file_path = '%s/*/*instanceIds.png' % directory
    isInstance=True
  else:
    file_path = '%s/*/*.png' % directory
    isInstance=False

  filenames = tf.gfile.Glob(file_path)
  filenames.sort()
  _process_image_files(name, filenames, num_shards, gray, isInstance=isInstance)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.train_shards')
  assert not FLAGS.test_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.test_shards')
  print('Saving results to %s' % FLAGS.output)

  # _process_dataset('train', FLAGS.train, FLAGS.train_shards, gray=False)
  # _process_dataset('train_labels_type', FLAGS.train_labels_type, FLAGS.train_shards)
  _process_dataset('train_labels_instance', FLAGS.train_labels_instance, FLAGS.train_shards)
  # _process_dataset('train_labels_disparity', FLAGS.train_labels_disparity, FLAGS.train_shards)

  if FLAGS.encode_test_images:
    # _process_dataset('test', FLAGS.test, FLAGS.test_shards, gray=False)
    # _process_dataset('test_labels_type', FLAGS.test_labels_type, FLAGS.test_shards)
    _process_dataset('test_labels_instance', FLAGS.test_labels_instance, FLAGS.test_shards)
    # _process_dataset('test_labels_disparity', FLAGS.test_labels_disparity, FLAGS.train_shards)

  if FLAGS.encode_val_images:
    # _process_dataset('val', FLAGS.val, FLAGS.test_shards, gray=False)
    # _process_dataset('val_labels_type', FLAGS.val_labels_type, FLAGS.test_shards)
    _process_dataset('val_labels_instance', FLAGS.val_labels_instance, FLAGS.test_shards)
    # _process_dataset('val_labels_disparity', FLAGS.val_labels_disparity, FLAGS.train_shards)

if __name__ == '__main__':
  tf.app.run()
