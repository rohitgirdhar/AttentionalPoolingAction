"""Provides data for the UCF101 dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import sys

from datasets import dataset_utils
from datasets.image_read_utils import _read_from_disk_spatial, \
    _decode_from_string, _read_from_disk_temporal, _get_frame_sublist, \
    _read_from_disk_pose
from tensorflow.python.platform import tf_logging as logging
from custom_ops.custom_ops_factory import read_json_pose, read_sparse_label

slim = tf.contrib.slim


class PreReadTextLineReader(tf.ReaderBase):
  @staticmethod
  def read(lines_queue):
    # just return the line from this queue.
    # The queue will be randomized if training and not if not.
    # Standard tf.TextLineReader will open the file and return line by line, we
    # don't want that, but want to randomize the whole file. Hence, this solves
    # this by first reading the whole file into the queue and then just picking
    # stuff from the queue.
    video_information = lines_queue.dequeue()
    return [video_information, video_information]  # make the video_info as the
                                                   # key for this datapoint as
                                                   # well


def decode_train_file_line(line, input_file_style='3-col',
                           input_file_style_label='one-label'):
  start_frame = 0
  if input_file_style == '3-col':
    fpath, nframes, label = tf.decode_csv(
        line, record_defaults=[[''], [-1], ['']],
        field_delim=' ')
  elif input_file_style == '4-col':
    fpath, start_frame, nframes, label = tf.decode_csv(
        line, record_defaults=[[''], [-1], [-1], ['']],
        field_delim=' ')
  else:
    raise ValueError('Unknown input file style: {0}'.format(
      input_file_style))

  if input_file_style_label == 'one-label':
    label = tf.string_to_number(label, out_type=tf.int32)
    label.set_shape(())
  elif input_file_style_label.startswith('multi-label'):
    nclasses = int(input_file_style_label[len('multi-label'):])
    label = read_sparse_label(label, nclasses)
    label.set_shape((nclasses,))
  return fpath, start_frame, nframes, label


def getReaderFn(num_samples, modality='rgb', dataset_dir='',
                randomFromSegmentStyle=None,
                input_file_style='3-col',
                input_file_style_label='one-label'):
  def readerFn():
    class reader_func(tf.ReaderBase):
      @staticmethod
      # def read(filename_queue):
      def read(value):
        # value = filename_queue.dequeue()
        fpath, start_frame, nframes, label = decode_train_file_line(
          value, input_file_style, input_file_style_label)
        # TODO(rgirdhar): Release the file_prefix='', file_zero_padding=4,
        # file_index=1 options to the bash script
        # TODO: Fix the optical_flow_frame number...
        optical_flow_frames = 1
        frame_sublist = _get_frame_sublist(0, nframes, num_samples,
                                           optical_flow_frames,
                                           randomFromSegmentStyle=randomFromSegmentStyle)
        # frame_sublist = tf.Print(frame_sublist, frame_sublist, "frame sublist:")
        if modality == 'rgb':
          assert(len(dataset_dir) >= 1)
          image_buffer = _read_from_disk_spatial(
              fpath, nframes, num_samples=num_samples,
              start_frame=start_frame,
              file_prefix='image', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0],
              frame_sublist=frame_sublist,
              randomFromSegmentStyle=randomFromSegmentStyle)
        elif modality.startswith('flow'):
          assert(len(dataset_dir) >= 1)
          optical_flow_frames = int(modality[4:])
          image_buffer = _read_from_disk_temporal(
              fpath, nframes, num_samples=num_samples,
              start_frame=start_frame,
              optical_flow_frames=optical_flow_frames,
              file_prefix='flow', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0],
              frame_sublist=frame_sublist,
              randomFromSegmentStyle=randomFromSegmentStyle)
        elif modality.startswith('rgb+flow'):
          assert(len(dataset_dir) >= 2)
          # in this case, fix the step for both the streams to ensure correspondence
          optical_flow_frames = int(modality[8:])
          rgb_image_buffer = _read_from_disk_spatial(
              fpath, nframes, num_samples=num_samples,
              start_frame=start_frame,
              file_prefix='image', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0],
              frame_sublist=frame_sublist)
          flow_image_buffer = _read_from_disk_temporal(
              fpath, nframes, num_samples=num_samples,
              start_frame=start_frame,
              optical_flow_frames=optical_flow_frames,
              file_prefix='flow', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[1],
              frame_sublist=frame_sublist)
          image_buffer = zip(rgb_image_buffer, flow_image_buffer)
          image_buffer = [[el[0]] + el[1] for el in image_buffer]
        elif modality.startswith('pose'):
          assert(len(dataset_dir) >= 1)
          if modality.startswith('posejson'):
            pose_frames = int(modality[8:])
            file_ext = '.json'
          elif modality.startswith('pose'):
            pose_frames = int(modality[4:])
            file_ext = '.jpg'
          image_buffer = _read_from_disk_pose(
              fpath, nframes, num_samples=num_samples,
              start_frame=start_frame,
              pose_frames=pose_frames,
              file_prefix='image', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0],
              frame_sublist=frame_sublist,
              randomFromSegmentStyle=randomFromSegmentStyle,
              file_ext=file_ext)
        else:
          logging.error('Unknown modality %s\n' % modality)
          raise ValueError()
        return [image_buffer, label, fpath, frame_sublist, start_frame]
    return reader_func
  return readerFn


def decoderFn(
  reader, num_samples=1, modality='rgb', dataset_dir='',
  randomFromSegmentStyle=True, num_pose_keypoints=16,
  pose_dataset_dir=None,
  num_object_catagories=80, objects_dataset_dir=None):
  class decoder_func(slim.data_decoder.DataDecoder):
    @staticmethod
    def list_items():
      return ['image', 'action_label', 'pose', 'im_ht', 'im_wd', 'objects']

    @staticmethod
    def decode(data, items):
      out = {}
      # Arguments:
      # data: Can be 3-col or 4-col CSV. A 3-col would look like "filepath
      # nframes class_id", 4-col will be similar for Charades like dataset
      # items: The different items to be returned.
      with tf.name_scope('decode_video'):
        if modality == 'rgb' or \
           modality.startswith('flow') or \
           modality.startswith('rgb+flow') or \
           modality.startswith('pose'):
          image_buffer, label, fpath, frame_sublist, start_frame = reader.read(data)
          # stacking required due to the way queues in main train loop work
          # image_buffer = tf.stack([tf.stack(_decode_from_string(el, modality)) for
          #                 el in image_buffer])
          image_lst = []
          image_hts = []
          image_wds = []
          for im_buf in image_buffer:
            temp = _decode_from_string(im_buf, modality)
            image_lst += temp[0]
            image_hts.append(temp[1])
            image_wds.append(temp[2])
          image_buffer = tf.stack(image_lst)
          im_ht = tf.reduce_max(image_hts)
          im_wd = tf.reduce_max(image_wds)
          # image_buffer = tf.stack([
          #   _decode_from_string(el, modality)[0] for el in image_buffer])
        else:
          logging.error('Unknown modality %s\n' % modality)
        # since my code gives a 0-1 image, change it back
        out['image'] = tf.cast(image_buffer * 255.0, tf.uint8)
        if 'pose' in items:
          if pose_dataset_dir is None:
            out['pose'] = [-tf.ones([num_pose_keypoints * 3,], dtype=tf.int64)]
          else:
            out['pose'] = [read_json_pose(tf.string_join([
              pose_dataset_dir, '/', fpath, '/',
              'image_',
              tf.as_string(frame_sublist_i+1, width=5, fill='0'),
              '.json'])) for frame_sublist_i in tf.unstack(frame_sublist)]
        if 'objects' in items:
          if objects_dataset_dir is None:
            out['objects'] = []
          else:
            out['objects'] = [tf.read_file(tf.string_join([
              objects_dataset_dir, '/', fpath, '/',
              'image_',
              tf.as_string(frame_sublist_i+1, width=5, fill='0'),
              '.txt'])) for frame_sublist_i in tf.unstack(frame_sublist)]
        out['action_label'] = label
        # The following is the original image size on disk,
        # on which pose etc would have been computed
        out['im_wd'] = tf.cast(im_wd, tf.int64)
        out['im_ht'] = tf.cast(im_ht, tf.int64)
        return [out[el] for el in items]
  return decoder_func


def count_frames_file(fpath, frameLevel=True):
  res = 0
  with open(fpath, 'r') as fin:
    for line in fin:
      if frameLevel:
        res += int(line.split()[1])
      else:
        res += 1
  return res


def gen_dataset(split_name, dataset_dir, file_pattern=None,
                reader=None,
                pose_dataset_dir=None,
                objects_dataset_dir=None,
                modality='rgb', num_samples=1,
                split_id=1, num_classes=0, list_fn=None,
                input_file_style='3-col',
                randomFromSegmentStyle=None, num_pose_keypoints=16,
                num_object_catagories=80,
                input_file_style_label='one-label'):
  """
  input_file_style_label: ['one-label'/'multi-label%d' % integer]
  """
  SPLITS_TO_SIZES = {
    'train': count_frames_file(list_fn('train', split_id), frameLevel=(num_samples==1)),
    'test': count_frames_file(list_fn('test', split_id), frameLevel=(num_samples==1)),
  }
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  _ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [? x ? x 3] color image.',
    'label': 'A single integer between 0 and %d' % num_classes,
  }
  LIST_FILE = list_fn(split_name, split_id)
  logging.info('Using file %s' % LIST_FILE)
  with open(LIST_FILE, 'r') as fin:
    data_sources = fin.read().splitlines()  # don't randomize here, in testing
                                            # I'll run without randomizing, and
                                            # the queue is going to randomize
                                            # automatically anyway

  # Allowing None in the signature so that dataset_factory can use the default.
  if not reader:
    reader = getReaderFn(num_samples, modality, [dataset_dir],
                         randomFromSegmentStyle, input_file_style,
                         input_file_style_label)

  labels_to_names = None
  # if dataset_utils.has_labels(dataset_dir):
  #   labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=data_sources,
      reader=lambda: PreReadTextLineReader,
      decoder=decoderFn(reader(), num_samples, modality, [dataset_dir],
                        randomFromSegmentStyle, num_pose_keypoints,
                        pose_dataset_dir,
                        num_object_catagories,
                        objects_dataset_dir),
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      labels_to_names=labels_to_names)
