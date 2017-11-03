import os
import json
from collections import OrderedDict
import numpy as np
import tensorflow as tf

cur_path = os.path.realpath(__file__)
ROOT_PATH = os.path.dirname(cur_path)

# add any new ops under the following
pose_to_heatmap_fn = tf.load_op_library(
  os.path.join(ROOT_PATH, 'pose_to_heatmap.so')).pose_to_heatmap
zero_out_channels_fn = tf.load_op_library(
  os.path.join(ROOT_PATH, 'zero_out_channels.so')).zero_out_channels
render_pose_fn = tf.load_op_library(
  os.path.join(ROOT_PATH, 'render_pose.so')).render_pose
render_objects_fn = tf.load_op_library(
  os.path.join(ROOT_PATH, 'render_objects.so')).render_objects

def pose_to_heatmap(*args, **kwargs):
  with tf.variable_scope('pose_to_heatmap_pyWrapper'):
    pose_img, pose_valid = pose_to_heatmap_fn(*args, **kwargs)
    out_channels = kwargs['out_channels']
    pose_img.set_shape((None, None, out_channels))
    pose_valid.set_shape((out_channels,))
    pose_img *= 255.0
    pose_img = tf.cast(pose_img, tf.uint8)
  return pose_img, pose_valid

def zero_out_channels(*args, **kwargs):
  with tf.variable_scope('zero_out_channels_pyWrapper'):
    return zero_out_channels_fn(*args, **kwargs)

def render_pose(*args, **kwargs):
  with tf.variable_scope('render_pose_pyWrapper'):
    out_channels = 3
    if kwargs['out_type'] == 'rgb':
      kwargs['out_type'] = 1
      out_channels = 3
    elif kwargs['out_type'] == 'split-channel':
      kwargs['out_type'] = 2
      out_channels = 18  # number of limbs
    img = render_pose_fn(*args, **kwargs)
    img *= 255.0
    img = tf.cast(img, tf.uint8)
    img.set_shape((None, None, out_channels))
  return img

# from render_pose.cc
mpii_to_coco = OrderedDict([
  (9, 0),
  (8, 1),
  (12, 2),
  (11, 3),
  (10, 4),
  (13, 5),
  (14, 6),
  (15, 7),
  (2, 8),
  (1, 9),
  (0, 10),
  (3, 11),
  (4, 12),
  (5, 13),
])
def read_json_pose_fn(fpath):
  try:
    with open(fpath, 'r') as fin:
      data = json.load(fin)
  except:
    print('Unable to open file {}'.format(fpath))
    return -np.ones((16*3,)).astype('int64')
  res = []
  for body in data['bodies']:
    mpii_joints = -np.ones((16, 3))
    joints = np.array(body['joints'])
    joints = np.reshape(joints, (-1, 3))
    joints[joints[..., :] <= 0] = -1
    mpii_joints[np.array(mpii_to_coco.keys()), :] = \
      joints[np.array(mpii_to_coco.values()), :]
    res += mpii_joints.reshape((-1,)).tolist()
  res = np.array(res).astype('int64')
  return res

def read_json_pose(*args):
  return tf.py_func(read_json_pose_fn, args, tf.int64)

def render_objects(*args, **kwargs):
  with tf.variable_scope('render_objects_pyWrapper'):
    img = render_objects_fn(*args, **kwargs)
    img *= 255.0
    img = tf.cast(img, tf.uint8)
    img.set_shape((None, None, kwargs['out_channels']))
  return img

def extract_glimpse(image, pose_label, orig_im_ht, orig_im_wd,
                    out_side, pad_ratio, parts_keep):
  # pose label is a [3x16xn,] vector
  # for now just take the first pose and crop out the human
  with tf.name_scope('ExtractGlimpse'):
    pose_label = pose_label[:16*3]
    pose_label = tf.reshape(pose_label, [16, 3])
    if len(parts_keep) > 0:
      pose_label = tf.gather(pose_label, parts_keep)
    if len(parts_keep) == 1:
      # now only one point, but need at least two to make a crop region
      delta = tf.to_int64(
        [tf.to_float(tf.shape(image)[-2]) * 0.1,
         tf.to_float(tf.shape(image)[-3]) * 0.1, 0])
      pose_label = tf.stack([
        pose_label[0] - delta, pose_label[0] + delta])
    pose_label_x = tf.to_float(pose_label[:, 0]) * \
        tf.to_float(tf.shape(image)[-2]) / tf.to_float(orig_im_wd)
    pose_label_y = tf.to_float(pose_label[:, 1]) * \
        tf.to_float(tf.shape(image)[-3]) / tf.to_float(orig_im_ht)
    pose_label = tf.stack([pose_label_y, pose_label_x])
    mx_pts = tf.to_int32(tf.reduce_max(pose_label, axis=1))
    mn_pts = tf.to_int32(tf.reduce_min(
      tf.where(tf.greater_equal(pose_label, 0), pose_label,
               tf.ones(pose_label.get_shape()) * 999999), axis=1))
    delta_0 = tf.to_int32(tf.to_float((mx_pts[0] - mn_pts[0])) * pad_ratio)
    delta_1 = tf.to_int32(tf.to_float((mx_pts[1] - mn_pts[1])) * pad_ratio)
    mx_pts = mx_pts + [delta_0, delta_1]
    mn_pts = mn_pts - [delta_0, delta_1]

    offset_ht = tf.maximum(mn_pts[0], 0)
    offset_wd = tf.maximum(mn_pts[1], 0)
    target_ht = tf.minimum(mx_pts[0]-offset_ht, tf.shape(image)[-3]-offset_ht-1)
    target_wd = tf.minimum(mx_pts[1]-offset_wd, tf.shape(image)[-2]-offset_wd-1)
    # image = tf.Print(image, [offset_ht, offset_wd, target_ht, target_wd,
    #                          tf.shape(image)], "stuff:")
    image = tf.cond(tf.logical_and(
      tf.greater(mx_pts[1], mn_pts[1]),
      tf.greater(mx_pts[0], mn_pts[0])),
      lambda: tf.image.crop_to_bounding_box(
        image, offset_ht, offset_wd, target_ht, target_wd),
      lambda: image)
    if out_side > 0:
      image = tf.image.resize_images(
        image, [out_side, out_side])
    return image

def read_sparse_label_fn(sparse_label, nclasses):
  """sparse_label is a string and return a 1D vector with the dense label
  """
  res = np.zeros((nclasses,), dtype='int32')
  res[np.array([int(el.split(':')[0]) for el in sparse_label.split(',')])] = \
      np.array([int(el.split(':')[1]) for el in sparse_label.split(',')])
  res[res < 0] = 0  # get rid of -1 label for now
  return res

def read_sparse_label(*args):
  return tf.py_func(read_sparse_label_fn, args, tf.int32)
