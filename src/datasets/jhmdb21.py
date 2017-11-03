"""Provides data for the JHMDB21 dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.video_data_utils import gen_dataset

_JHMDB21_TRINITY_LIST_DIR = '/data/rgirdhar/Data2/Projects/2016/002_VideoRepresentation/StandardData/002_JHMDB/Processed/Lists/train_test_lists/'
_JHMDB21_TRINITY_POSE_LABEL_DIR = '/data/rgirdhar/Data2/Projects/2016/002_VideoRepresentation/StandardData/002_JHMDB/Processed/Features/001_CPM_Pose/'
_JHMDB21_TRINITY_OBJECTS_LABEL_DIR = ''

def get_split(split_name, dataset_dir,
              file_pattern=None,
              reader=None, **kwargs):
              # dataset_list_dir=_JHMDB21_TRINITY_LIST_DIR,
              # modality='rgb', num_samples=1,
              # split_id=1, **kwargs):

  _NUM_CLASSES = 21
  # There are no pose labels, but need to keep this to load models from MPII
  # trained
  # Also, now the processing can still avoided by having no loss on pose
  _NUM_POSE_KEYPOINTS = 16
  _LIST_FN = lambda split, id: \
      '%s/%s_split%d.txt' % (
        kwargs['dataset_list_dir'] if 'dataset_list_dir' in kwargs
        else _JHMDB21_TRINITY_LIST_DIR,
        split, id)

  kwargs['num_pose_keypoints'] = _NUM_POSE_KEYPOINTS
  kwargs['num_classes'] = _NUM_CLASSES
  kwargs['list_fn'] = _LIST_FN
  return gen_dataset(split_name, dataset_dir, file_pattern,
                     reader,
                     pose_dataset_dir=_JHMDB21_TRINITY_POSE_LABEL_DIR,
                     objects_dataset_dir=_JHMDB21_TRINITY_OBJECTS_LABEL_DIR,
                     **kwargs), _NUM_POSE_KEYPOINTS
                     # modality, num_samples, split_id,
                     # _NUM_CLASSES, _LIST_FN, **kwargs), _NUM_POSE_KEYPOINTS
