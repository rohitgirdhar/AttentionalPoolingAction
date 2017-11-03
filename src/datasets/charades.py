"""Provides data for the HMDB51 dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.video_data_utils import gen_dataset
import tensorflow as tf

_CHARADES_TRINITY_LIST_DIR = '/data/rgirdhar/Data2/Projects/2016/002_VideoRepresentation/StandardData/001_Charades/v1/Lists/train_test_lists/'
_CHARADES_TRINITY_POSE_LABEL_DIR = '/scratch/rgirdhar/Datasets/Video/004_Charades/Processed/002_Pose_CPM_v2/'

def get_split(split_name, dataset_dir,
              file_pattern=None,
              reader=None, **kwargs):

  _NUM_CLASSES = 157
  # There are no pose labels, but need to keep this to load models from MPII
  # trained
  # Also, now the processing can still avoided by having no loss on pose
  _NUM_POSE_KEYPOINTS = 16
  # Need to do this otherwise the lambda function defined below will not work
  # It evaluates the kwargs['..'] also when evaluated
  if 'dataset_list_dir' not in kwargs:
    dataset_list_dir = _CHARADES_TRINITY_LIST_DIR
  else:
    dataset_list_dir = kwargs['dataset_list_dir']
  _LIST_FN = lambda split, id: \
      '%s/%s_split%d.txt' % (
        dataset_list_dir,
        split, id)

  kwargs['num_pose_keypoints'] = _NUM_POSE_KEYPOINTS
  kwargs['num_classes'] = _NUM_CLASSES
  kwargs['list_fn'] = _LIST_FN
  with open(_LIST_FN(split_name, kwargs['split_id']), 'r') as fin:
    ncols = len(fin.readline().strip().split())
  if ncols == 4:
    input_file_style = '4-col'
  elif ncols == 3:
    input_file_style = '3-col'  # since video level testing with mAP
  else:
    raise ValueError('Invalid file style')
  tf.logging.info('Using input_file_style {}'.format(input_file_style))

  # need to remove some things from kwargs (if they exist) before passing on
  kwargs.pop('dataset_list_dir', [])
  return gen_dataset(split_name, dataset_dir, file_pattern,
                     reader,
                     pose_dataset_dir=_CHARADES_TRINITY_POSE_LABEL_DIR,
                     input_file_style=input_file_style,
                     **kwargs), _NUM_POSE_KEYPOINTS
