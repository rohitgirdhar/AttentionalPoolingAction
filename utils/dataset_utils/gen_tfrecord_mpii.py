from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import scipy.io
import operator
import numpy as np

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Set the following paths
_MPII_MAT_FILE = '/path/to/mpii_human_pose_v1_u12_1.mat'
_IMG_DIR = '/path/to/MPII/images/'


dataset_dir = '../../src/data/mpii/mpii_tfrecords/'
_SPLITS_PATH = '../../src/data/mpii/lists/'

# Seed for repeatability.
_RANDOM_SEED = 42

# The number of shards per dataset split.
_NUM_SHARDS = 20

_NUM_JOINTS = 16  # for pose

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width,
                       pose, # [x,y,is_vis,...]
                       action_label):
  assert(len(pose) % (_NUM_JOINTS * 3) == 0)
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/pose': int64_feature([int(el) for el in pose]),
      'image/class/action_label': int64_feature(action_label),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'mpii_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, list_to_write, dataset_dir):
  num_per_shard = int(math.ceil(len(list_to_write) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(list_to_write))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(list_to_write), shard_id))
            sys.stdout.flush()

            # Read the filename:
            fname = os.path.join(_IMG_DIR, list_to_write[i][0])
            action_label = list_to_write[i][1]
            poses = list_to_write[i][2]
            all_joints = []
            for pose in poses:
              joints = dict((el[0], [el[1], el[2], el[3]]) for el in pose)
              final_pose = []
              for i in range(_NUM_JOINTS):
                if i in joints:
                  final_pose.append(joints[i])
                else:
                  final_pose.append([-1, -1, 0])
              final_pose = [item for sublist in final_pose for item in sublist]
              all_joints += final_pose
            assert(len(all_joints) % 16 == 0)
            image_data = tf.gfile.FastGFile(fname, 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            example = image_to_tfexample(
                image_data, 'jpg', height, width, all_joints, action_label)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _get_action_class(cname, D, act_id):
  try:
    if cname not in D:
      D[cname] = (len(D.keys()), set([act_id]))  # act_id is the actual MPII action id
    else:
      D[cname][1].add(act_id)
      # It's pretty crazy that same action will have multiple action IDs
    return D[cname][0]
  except Exception, e:
    print('Invalid class name {}. setting -1. {}'.format(cname, e))
    return -1


def main():
  T = scipy.io.loadmat(_MPII_MAT_FILE, squeeze_me=True,
                       struct_as_record=False)
  annots = T['RELEASE'].annolist
  is_train = T['RELEASE'].img_train
  action_label = T['RELEASE'].act
  splits = ['train', 'val', 'test']
  lists_to_write = {}
  img_id_in_split = {}
  all_imnames = []
  for spl in splits:
    lists_to_write[spl] = []
    img_id_in_split[spl] = []
  splits_filenames = {}
  filename_to_split = {}
  actclassname_to_id = {}
  for spl in splits:
    with open(_SPLITS_PATH.format(spl), 'r') as fin:
      splits_filenames[spl] = fin.read().splitlines()
      filename_to_split.update(dict(zip(
        splits_filenames[spl], [spl] * len(splits_filenames[spl]))))
  for aid,annot in enumerate(annots):
    imname = annot.image.name
    all_imnames.append(imname)
    try:
      this_split = filename_to_split[imname[:-4]]
    except:
      continue  # ignore this image
    points_fmted = []  # put all points one after the other
    if 'annorect' in dir(annot):
      all_rects = annot.annorect
      if isinstance(all_rects, scipy.io.matlab.mio5_params.mat_struct):
        all_rects = np.array([all_rects])
      for rect in all_rects:
        points_rect = []
        try:
          points = rect.annopoints.point
        except:
          continue
        if isinstance(points, scipy.io.matlab.mio5_params.mat_struct):
          points = np.array([points])
        for point in points:
          try:
            is_visible = point.is_visible if point.is_visible in [1,0] else 0
          except:
            is_visible = 0
          points_rect.append((point.id, point.x, point.y, is_visible))
        points_fmted.append(points_rect)
    [el.sort() for el in points_fmted]

    # the following assert is not true, so putting -1 when writing it out
    # assert(all([len(el) == 16 for el in points_fmted]))
    image_obj = (annot.image.name,
                 _get_action_class(action_label[aid].act_name,
                                   actclassname_to_id,
                                   action_label[aid].act_id),
                 points_fmted)
    if os.path.exists(os.path.join(_IMG_DIR, imname)):
      lists_to_write[this_split].append(image_obj)
      img_id_in_split[this_split].append(aid+1)  # 1-indexed
  cls_ids = sorted(actclassname_to_id.items(), key=operator.itemgetter(1))
  print('Total classes found: {}'.format(len(cls_ids)))
  #write out the dictionary of classnames
  with open(os.path.join(dataset_dir, 'classes.txt'), 'w') as fout:
    fout.write('\n'.join([el[0] + ';' + ','.join([
      str(e) for e in list(el[1][1])]) for el in cls_ids]))

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  # Only randomize the train set
  random.seed(_RANDOM_SEED)
  train_ids = range(len(lists_to_write['train']))
  random.shuffle(train_ids)
  lists_to_write['train'] = [lists_to_write['train'][i] for i in train_ids]
  img_id_in_split['train'] = [img_id_in_split['train'][i] for i in train_ids]

  with open(os.path.join(dataset_dir, 'imnames.txt'), 'w') as fout:
    fout.write('\n'.join(all_imnames))
  for spl in splits:
    with open(os.path.join(
      dataset_dir, '{}_ids.txt'.format(spl)), 'w') as fout:
      fout.write('\n'.join([str(el) for el in img_id_in_split[spl]]))
    spl_name = spl
    if spl in ['train', 'val']:
      spl_name = 'trainval_' + spl  # would be useful when training on tr+val
    print('Writing {} images for split {}.'.format(
      len(lists_to_write[spl]), spl))
    _convert_dataset(spl_name, lists_to_write[spl],
                     dataset_dir)

  print('\nFinished converting the MPII dataset!')

if __name__ == '__main__':
  main()
