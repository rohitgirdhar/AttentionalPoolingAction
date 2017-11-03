"""Provides data for the HMDB51 dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.video_data_utils import gen_dataset

_HICO_TRINITY_LIST_DIR = '/data/rgirdhar/Data2/Projects/2016/002_VideoRepresentation/StandardData/005_HICO/data_videoFormat/001_Basic/train_test_lists'
_HICO_TRINITY_POSE_LABEL_DIR = '/scratch/rgirdhar/Datasets/Image/003_HICO/data_videoFormat/001_Basic/features/001_CPMPose/'
_HICO_DATASET_DIR = '/scratch/rgirdhar/Datasets/Image/003_HICO/data_videoFormat/001_Basic/frames'

def get_split(split_name, dataset_dir,
              file_pattern=None,
              reader=None, **kwargs):

  _NUM_CLASSES = 600
  # There are no pose labels, but need to keep this to load models from MPII
  # trained
  # Also, now the processing can still avoided by having no loss on pose
  _NUM_POSE_KEYPOINTS = 16
  if 'dataset_list_dir' not in kwargs:
    dataset_list_dir = _HICO_TRINITY_LIST_DIR
  else:
    dataset_list_dir = kwargs['dataset_list_dir']
  _LIST_FN = lambda split, id: \
      '%s/%s_split%d.txt' % (
        dataset_list_dir,
        split, id)

  kwargs['num_pose_keypoints'] = _NUM_POSE_KEYPOINTS
  kwargs['num_classes'] = _NUM_CLASSES
  kwargs['list_fn'] = _LIST_FN
  input_file_style = '3-col'
  kwargs.pop('dataset_list_dir', [])
  return gen_dataset(split_name, dataset_dir,
                     file_pattern, reader,
                     pose_dataset_dir=_HICO_TRINITY_POSE_LABEL_DIR,
                     input_file_style=input_file_style,
                     **kwargs), _NUM_POSE_KEYPOINTS
