from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = 'mpii_%s_*.tfrecord'

SPLITS_TO_SIZES = {'trainval_train': 8219, 'trainval_val': 6988,
                   'trainval': 15207,  # 8219 + 6988
                   'test': 5709}

_NUM_CLASSES = 393  # activities

_NUM_POSE_KEYPOINTS = 16

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A pose representation, [x1,y1,is_visible1,...]',
}

def _tfrecord_file_pattern_to_list(pattern):
  res = glob.glob(pattern)
  return sorted(res)


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  # The following is important to ensure the files are read in order, because
  # otherwise test time output can be generated in any random order
  file_pattern = _tfrecord_file_pattern_to_list(file_pattern)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/pose': tf.VarLenFeature(dtype=tf.int64),
      'image/class/action_label': tf.FixedLenFeature(
        (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature(
        (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width': tf.FixedLenFeature(
        (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'pose': slim.tfexample_decoder.Tensor('image/class/pose'),
      'action_label': slim.tfexample_decoder.Tensor('image/class/action_label'),
      'im_ht': slim.tfexample_decoder.Tensor('image/height'),
      'im_wd': slim.tfexample_decoder.Tensor('image/width'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names), _NUM_POSE_KEYPOINTS
