"""Config System
"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


#
# Input options
#


__C.INPUT = edict()

# normal: normal image
# rendered-pose: rendered pose on black bg
# rendered-pose-on-image: rendered onto the image
__C.INPUT.INPUT_IMAGE_FORMAT = 'normal'

# pose renders can be 'rgb' or 'split-channel'
__C.INPUT.INPUT_IMAGE_FORMAT_POSE_RENDER_TYPE = 'rgb'

# input glimpse options
__C.INPUT.POSE_GLIMPSE_CONTEXT_RATIO = 0.0  # ratio of glimpse area to pad around
# set the following to true to resize the output to [IMAGE_SIZE, IMAGE_SIZE]
# square
__C.INPUT.POSE_GLIMPSE_RESIZE = False
# list part sof the pose to keep in glimpse. Empty => all parts to keep
__C.INPUT.POSE_GLIMPSE_PARTS_KEEP = []


__C.INPUT.SPLIT_ID = 1  # for dataset with multiple splits (hmdb)

# FOR VIDEO
__C.INPUT.VIDEO = edict()
__C.INPUT.VIDEO.MODALITY = 'rgb'  # rgb/flow5/flow10 etc

#
# Training options
#

__C.TRAIN = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 10

__C.TRAIN.WEIGHT_DECAY = 0.0005

# set to a positive value to clip the gradients at that l2 norm
__C.TRAIN.CLIP_GRADIENTS = -1.0

# the following should have been in the INPUT, but are here for historical
# reasons
__C.TRAIN.IMAGE_SIZE = 450  # final cropped image size
__C.TRAIN.RESIZE_SIDE = 480  # resize the input image to this size for preproc
## The RESIZE_SIDE is the size for the smallest side, so be careful,
## MPII has images with extreme ratios
## Note that if the difference RESIZE_SIDE to IMAGE_SIZE is too high,
## most of the image being fed into the network will be small parts of the
## image

# This is the side of the heatmap before putting into queues
# Ideally, resize it to the final target size so that there is no
# need for a resize before computing loss. For inception-v2 with 450 input, the
# output is 15x15
__C.TRAIN.FINAL_POSE_HMAP_SIDE = 15

__C.TRAIN.LABEL_SMOOTHING = False

__C.TRAIN.MOVING_AVERAGE_VARIABLES = None

__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.LEARNING_RATE_DECAY_RATE = 0.33
__C.TRAIN.END_LEARNING_RATE = 0.00001

__C.TRAIN.NUM_STEPS_PER_DECAY = 0  # if this is not 0, the NUM_EPOCHS_PER_DECAY
                                   # is ignored and this is used
__C.TRAIN.NUM_EPOCHS_PER_DECAY = 40.0

__C.TRAIN.LEARNING_RATE_DECAY_TYPE = 'exponential'


__C.TRAIN.OPTIMIZER = 'momentum'
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.ADAM_BETA1 = 0.9
__C.TRAIN.ADAM_BETA2 = 0.999
__C.TRAIN.OPT_EPSILON = 1.0

__C.TRAIN.TRAINABLE_SCOPES = ''

__C.TRAIN.MAX_NUMBER_OF_STEPS = 100000

__C.TRAIN.LOG_EVERY_N_STEPS = 10

__C.TRAIN.SAVE_SUMMARIES_SECS = 300

__C.TRAIN.SAVE_INTERVAL_SECS = 1800

__C.TRAIN.IGNORE_MISSING_VARS = True

__C.TRAIN.CHECKPOINT_PATH = 'data/pretrained_models/inception_v3.ckpt'

# __C.TRAIN.CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits,PoseLogits'
__C.TRAIN.CHECKPOINT_EXCLUDE_SCOPES = ''

__C.TRAIN.DATASET_SPLIT_NAME = 'trainval_train'

# loss fn can be from the list or empty '', i.e. no loss on that modality
__C.TRAIN.LOSS_FN_POSE = 'l2'  # can be 'l2'/'log-loss'/'sigmoid-log-loss'/'cosine-loss'
__C.TRAIN.LOSS_FN_POSE_WT = 1.0
__C.TRAIN.LOSS_FN_POSE_SAMPLED = False  # Harder loss, sample the negatives
__C.TRAIN.LOSS_FN_ACTION = 'softmax-xentropy'  # can be 'softmax-xentropy'
__C.TRAIN.LOSS_FN_ACTION_WT = 1.0

__C.TRAIN.VAR_NAME_MAPPER = ''  # to be used when loading from npy checkpoints
                                # see options in restore/var_name_mapper.py

__C.TRAIN.VIDEO_FRAMES_PER_VIDEO = 1

# If true, divide the video into segments and read
# a random frame from that segment
__C.TRAIN.READ_SEGMENT_STYLE = False

__C.TRAIN.ITER_SIZE = 1  # accumulate gradients over this many iterations

__C.TRAIN.OTHER_IMG_SUMMARIES_TO_ADD = ['PosePrelogitsBasedAttention']

#
# Testing options
#

__C.TEST = edict()

__C.TEST.BATCH_SIZE = 10

__C.TEST.DATASET_SPLIT_NAME = 'trainval_val'

__C.TEST.MAX_NUM_BATCHES = None

__C.TEST.CHECKPOINT_PATH = b''

__C.TEST.MOVING_AVERAGE_DECAY = None

__C.TEST.VIDEO_FRAMES_PER_VIDEO = 1  # single image dataset. Set 25 for hmdb

__C.TEST.EVAL_METRIC = ''  # normal eval. Set ='mAP' to compute that.


#
# Network properties
#

__C.NET = edict()
# The following replaces the action logits with one computed by weighting the
# output using pose heatmaps
__C.NET.USE_POSE_ATTENTION_LOGITS = False
__C.NET.USE_POSE_ATTENTION_LOGITS_DIMS = [-1]  # by default use all parts
# set following true to have a heatmap as the avg of all heatmaps
__C.NET.USE_POSE_ATTENTION_LOGITS_AVGED_HMAP = False


# The following will replace the action logits with one computed over the last
# pose logits
__C.NET.USE_POSE_LOGITS_DIRECTLY = False
# set true to also have the actual logits concatenated to the output
__C.NET.USE_POSE_LOGITS_DIRECTLY_PLUS_LOGITS = False
# Another version, after talking to Deva on March 20, 2017. Concat before avg
# pool and remove the extra layer.
# The following by default contain the image logits
__C.NET.USE_POSE_LOGITS_DIRECTLY_v2 = False
__C.NET.USE_POSE_LOGITS_DIRECTLY_v2_EXTRA_LAYER = False

# The following will replace the action logits with a one computed using an
# unconstrained attention predictor based on the pose output
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION = False
# REMOVED THIS TO DEPRECATE
# # setting the following to true basically just reproduces the original system
# # (doesnot use any attention). I just used it to debug that this can reproduce
# # the original numbers (nothing else got screwed up)
# __C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_DEBUG = False
# set the following to more to have more layers predicting the unconstrained
# attention map
# DEPRECATING the following, commented out for now, will be removed later.
# __C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_NLAYERS = 1
# set True to enforce the attention map that is learnt to be passed  through a
# spatial softmax
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_SOFTMAX_ATT = False
# Pass the attention through a relu
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_RELU_ATT = False
# 21 April 2017: This is not DEPRECATED because it didn't help, so it won't
# work with code now. This was to simplify code for TopDownAttention endpoint
# # Create an attention map for each class
# adding it again on July 26, 2017 for NIPS17 rebuttal
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_PER_CLASS = False
# Train attention directly over image features
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_SINGLE_LAYER_ATT = False
# Add the predicted pose to the logits features
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT = False
# 2-layers over the pose logits
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT_2LAYER = False
# Allow for Rank > 1 approximation. Other options might not work with this
__C.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_RANK = 1

# Do attention on temporal pooling as well
__C.NET.USE_TEMPORAL_ATT = False

# Bilinear pooling baselines
__C.NET.USE_COMPACT_BILINEAR_POOLING = False

# Set which endpoint serves as the output for pose
__C.NET.LAST_CONV_MAP_FOR_POSE = edict()
__C.NET.LAST_CONV_MAP_FOR_POSE.inception_v2_tsn = 'InceptionV2_TSN/inception_5a'
__C.NET.LAST_CONV_MAP_FOR_POSE.inception_v3 = 'Mixed_7c'
__C.NET.LAST_CONV_MAP_FOR_POSE.resnet_v1_101 = 'resnet_v1_101/block4'
__C.NET.LAST_CONV_MAP_FOR_POSE.vgg_16 = 'vgg_16/conv5'


# Train the top BN. Useful when training flow/multi-channel inputs other than
# RGB. In case of ResNet, this means "train only top_bn", and keep others
# fixed.
__C.NET.TRAIN_TOP_BN = False
# Dropout
# -1 (<0) => Use the network default. Else, use this value
__C.NET.DROPOUT = -1.0

#
# MISC
#

# For reproducibility
__C.RNG_SEED = 42

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.EXP_DIR = 'expt_outputs/'

__C.DATASET_NAME = 'mpii'

__C.DATASET_DIR = 'data/mpii/mpii_tfrecords'

# Set the following if using the train_test files from non-std location
__C.DATASET_LIST_DIR = ''

__C.MODEL_NAME = 'inception_v3'

__C.NUM_READERS = 4

__C.NUM_PREPROCESSING_THREADS = 4

__C.GPUS = '2'

__C.HEATMAP_MARKER_WD_RATIO = 0.1

__C.MAX_INPUT_IMAGE_SIZE = 512  # to avoid arbitrarily huge input images

# ['one-label'/'multi-label%d']
__C.INPUT_FILE_STYLE_LABEL = ''


def get_output_dir(config_file_name):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.EXP_DIR, osp.basename(config_file_name)))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
