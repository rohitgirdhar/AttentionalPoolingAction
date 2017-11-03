# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import numpy as np
import sys

from nets import alexnet
from nets import cifarnet
from nets import inception
from nets import lenet
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg

sys.path.append('libs/tensorflow_compact_bilinear_pooling/')
from compact_bilinear_pooling import compact_bilinear_pooling_layer

slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'overfeat': overfeat.overfeat,
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v1': inception.inception_v1,
                'inception_v2': inception.inception_v2,
                'inception_v2_tsn': inception.inception_v2_tsn,
                'inception_v3': inception.inception_v3,
                'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'lenet': lenet.lenet,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                'resnet_v1_200': resnet_v1.resnet_v1_200,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
               }

last_conv_map = {'inception_v3': 'Mixed_7c',
                 'inception_v2_tsn': 'InceptionV2_TSN/inception_5b',
                 'resnet_v1_101': 'resnet_v1_101/block4',
                 'vgg_16': 'vgg_16/conv5',
                }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v1': inception.inception_v3_arg_scope,
                  'inception_v2': inception.inception_v3_arg_scope,
                  'inception_v2_tsn': inception.inception_v2_tsn_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2':
                  inception.inception_resnet_v2_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                 }


def get_network_fn(name, num_classes, num_pose_keypoints, cfg,
                   weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    num_pose_keypoints: The number of channels to output for pose.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      frames_per_video = 1  # same for single image datasets
      if images.get_shape().ndims == 5:
        im_shape = images.get_shape().as_list()
        frames_per_video = im_shape[1]
        images = tf.reshape(
          images, [-1, im_shape[-3], im_shape[-2], im_shape[-1]])

      # Main Network Function
      kwargs = {}
      if cfg.NET.DROPOUT >= 0:  # if -1, then just ignore it and use nw def.
        kwargs['dropout_keep_prob'] = (1-cfg.NET.DROPOUT)
      logits, end_points = func(images, num_classes, is_training=is_training,
                                train_top_bn=cfg.NET.TRAIN_TOP_BN,
                                **kwargs)

      # rgirdhar: add another end point for heatmap prediction
      try:
        last_conv = end_points[last_conv_map[name]]
      except:
        raise ValueError('End point {} not found. Choose from: {}'.format(
          last_conv_map[name], ' '.join(end_points)))
      random_normal = lambda stddev: tf.random_normal_initializer(0.0, stddev)

      with slim.arg_scope([slim.dropout],
                          is_training=is_training,
                          keep_prob=0.2 if cfg.NET.DROPOUT < 0
                                        else (1.0-cfg.NET.DROPOUT)):
        with tf.variable_scope('PoseLogits'):
          last_conv_pose_name = getattr(
            cfg.NET.LAST_CONV_MAP_FOR_POSE, name)
          last_conv_pose = end_points[last_conv_pose_name]
          pose_pre_logits = slim.conv2d(
            last_conv_pose, 768, [1, 1],
            weights_initializer=random_normal(0.001),
            activation_fn=tf.nn.relu,
            normalizer_fn=None,
            biases_initializer=tf.zeros_initializer(),
            padding='SAME', scope='ExtraConv2d_1x1')
          pose_logits = slim.conv2d(pose_pre_logits, num_pose_keypoints, [1, 1], activation_fn=None,
                                    normalizer_fn=None, scope='Conv2d_1c_1x1')
          end_points['PoseLogits'] = pose_logits

        if cfg.NET.USE_POSE_ATTENTION_LOGITS:
          with tf.variable_scope('PoseAttention'):
            # use the pose prediction as an attention map to get the features
            # step1: split pose logits over channels
            pose_logits_parts = tf.split(
              pose_logits, pose_logits.get_shape().as_list()[-1],
              axis=pose_logits.get_shape().ndims-1)
            part_logits = []
            # allows to choose which dimension of pose to use for heatmaps
            parts_to_use = pose_logits_parts
            if cfg.NET.USE_POSE_ATTENTION_LOGITS_DIMS != [-1]:
              parts_to_use = (np.array(pose_logits_parts)[
                cfg.NET.USE_POSE_ATTENTION_LOGITS_DIMS]).tolist()
            tf.logging.info('Using {} parts for pose attention logits'.format(
              len(parts_to_use)))
            for part in parts_to_use:
              part_logits.append(tf.reduce_mean(part * last_conv, axis=[1,2],
                                                keep_dims=True))
            if cfg.NET.USE_POSE_ATTENTION_LOGITS_AVGED_HMAP:
              part_logits.append(tf.reduce_mean(
                last_conv * tf.reduce_mean(pose_logits, axis=-1, keep_dims=True),
                axis=[1,2], keep_dims=True))
            part_logits.append(tf.reduce_mean(last_conv, axis=[1,2],
                                              keep_dims=True))
            net = tf.concat(part_logits, axis=-1)
            net = slim.dropout(net)
            logits = slim.conv2d(net, num_classes, [1, 1],
                                 weights_initializer=random_normal(0.001),
                                 biases_initializer=tf.zeros_initializer(),
                                 activation_fn=None,
                                 normalizer_fn=None)
        elif cfg.NET.USE_POSE_LOGITS_DIRECTLY:
          with tf.variable_scope('ActionFromPose'):
            net = tf.reduce_mean(
              pose_pre_logits, axis=[1, 2], keep_dims=True)
            net = slim.conv2d(net, 768, [1, 1],
                              normalizer_fn=None,
                              weights_initializer=random_normal(0.001),
                              biases_initializer=tf.zeros_initializer())
            if cfg.NET.USE_POSE_LOGITS_DIRECTLY_PLUS_LOGITS:
              net = tf.concat([
                net, tf.reduce_mean(last_conv, axis=[1, 2], keep_dims=True)],
                axis=-1)
            net = slim.dropout(net)
            logits = slim.conv2d(net, num_classes, [1, 1],
                                 weights_initializer=random_normal(0.001),
                                 biases_initializer=tf.zeros_initializer(),
                                 activation_fn=None,
                                 normalizer_fn=None)
        elif cfg.NET.USE_POSE_LOGITS_DIRECTLY_v2:
          with tf.variable_scope('ActionFromPose_v2'):
            net = tf.concat([
              pose_pre_logits,
              last_conv],
              axis=-1)
            if cfg.NET.USE_POSE_LOGITS_DIRECTLY_v2_EXTRA_LAYER:
              net = tf.nn.relu(net)
              net = slim.conv2d(net, net.get_shape().as_list()[-1], [1, 1],
                                weights_initializer=random_normal(0.001),
                                biases_initializer=tf.zeros_initializer())
            net = tf.reduce_mean(net, axis=[1, 2], keep_dims=True)
            net = slim.dropout(net)
            logits = slim.conv2d(net, num_classes, [1, 1],
                                 weights_initializer=random_normal(0.001),
                                 biases_initializer=tf.zeros_initializer(),
                                 activation_fn=None,
                                 normalizer_fn=None)
        elif cfg.NET.USE_COMPACT_BILINEAR_POOLING:
          last_conv_shape = last_conv.get_shape().as_list()
          net = compact_bilinear_pooling_layer(
            last_conv, last_conv, last_conv_shape[-1])
          net.set_shape([last_conv_shape[0], last_conv_shape[-1]])
          net = tf.expand_dims(tf.expand_dims(
            net, 1), 1)
          net = slim.dropout(net)
          logits = slim.conv2d(net, num_classes, [1, 1],
                               weights_initializer=random_normal(0.001),
                               biases_initializer=tf.zeros_initializer(),
                               activation_fn=None,
                               normalizer_fn=None)
        elif cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION:
          with tf.variable_scope('PosePrelogitsBasedAttention'):
            # If the following is set, just train on top of image features,
            # don't add the prelogits at all. This was useful as pose seemed to
            # not help with it at all.
            if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_SINGLE_LAYER_ATT:
              net = last_conv
            else:
              net = pose_pre_logits
            # nMaps = num_classes if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_PER_CLASS else 1
            # For simplicity, since multiple maps doesn't seem to help, I'm
            # not allowing that to keep the following code simple.
            # nMaps = 1
            # For NIPS2017 rebuttal, they wanted to see nums with per-class
            # attention, so doing that too
            nMaps = num_classes if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_PER_CLASS else 1
            all_att_logits = []
            for rank_id in range(cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_RANK):
              scope_name = 'Conv2d_PrePose_Attn'
              if rank_id >= 1:
                scope_name += str(rank_id)
              net = slim.conv2d(net, nMaps,
                                [1, 1],
                                weights_initializer=random_normal(0.001),
                                biases_initializer=tf.zeros_initializer(),
                                activation_fn=None,
                                normalizer_fn=None,
                                scope=scope_name)
              all_att_logits.append(net)
            if len(all_att_logits) > 1:
              attention_logits = tf.stack(all_att_logits, axis=-1)
            else:
              attention_logits = all_att_logits[0]

            if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_SOFTMAX_ATT:
              # bring the number of channels earlier to make softmax easier
              attention_logits = tf.transpose(attention_logits, [0, 3, 1, 2])
              att_shape = attention_logits.get_shape().as_list()
              attention_logits = tf.reshape(
                attention_logits, [att_shape[0], att_shape[1], -1])
              attention_logits = tf.nn.softmax(attention_logits)
              attention_logits = tf.reshape(attention_logits, att_shape)
              attention_logits = tf.transpose(attention_logits, [0, 2, 3, 1])
            if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_RELU_ATT:
              attention_logits = tf.nn.relu(attention_logits)
            end_points['PosePrelogitsBasedAttention'] = attention_logits

            if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT:
              if cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT_2LAYER:
                pose_logits = slim.conv2d(
                  pose_logits, pose_logits.get_shape()[-1],
                  [1, 1], weights_initializer=random_normal(0.001),
                  biases_initializer=tf.zeros_initializer())
              last_conv = tf.concat([last_conv, pose_logits], axis=-1)
            last_conv = slim.dropout(last_conv)
            # Top-down attention
            all_logits = []
            for _ in range(cfg.NET.USE_POSE_PRELOGITS_BASED_ATTENTION_RANK):
              logits = slim.conv2d(last_conv, num_classes, [1, 1],
                                   weights_initializer=random_normal(0.001),
                                   biases_initializer=tf.zeros_initializer(),
                                   activation_fn=None, normalizer_fn=None)
              all_logits.append(logits)
            if len(all_logits) > 1:
              logits = tf.stack(all_logits, axis=-1)
            else:
              logits = all_logits[0]
            end_points['TopDownAttention'] = logits

            # attended_feats = []
            # for attention_logit in tf.unstack(attention_logits, axis=-1):
            #   attended_feats.append(tf.reduce_mean(
            #     tf.expand_dims(attention_logit, axis=-1) * logits,
            #     axis=[1,2],
            #     keep_dims=True))
            # attended_feat = tf.stack(attended_feats, axis=-1)
            # # Since only 1 attention map (asserted above)
            # logits = attended_feat[..., 0]

            # better way to do the above:
            logits = tf.reduce_mean(
              attention_logits * logits,
              axis=[1, 2],
              keep_dims=True)
            if logits.get_shape().ndims == 5:
              # i.e. rank was > 1
              logits = tf.reduce_sum(logits, axis=-1)

            # if nMaps == 1:
            #   # remove the extra dimension that is added for multi-class
            #   # attention case
            #   attended_feat = attended_feat[..., 0]
            #   logits = slim.conv2d(attended_feat, num_classes, [1, 1],
            #                        weights_initializer=random_normal(0.001),
            #                        biases_initializer=tf.zeros_initializer(),
            #                        activation_fn=None,
            #                        normalizer_fn=None)
            # else:
            #   logits = tf.concat([
            #     slim.conv2d(el, 1, [1, 1],
            #                 weights_initializer=random_normal(0.001),
            #                 biases_initializer=tf.zeros_initializer(),
            #                 activation_fn=None,
            #                 normalizer_fn=None) for el in
            #     tf.unstack(attended_feat, axis=-1)], axis=-1)
        # This is just to protect against the case where I don't do any of the
        # above and get the original logits from the network, which has already
        # been squeezed, or in case of vgg 16, passed through fc layers
        if logits.get_shape().ndims > 2:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits

      if frames_per_video > 1:
        with tf.name_scope('FramePooling'):
          # for now stick with avg pool
          end_points['logits_beforePool'] = logits
          old_logits = logits
          logits = tf.stack([el for el in tf.split(
            old_logits, int(old_logits.get_shape().as_list()[0] /
                            frames_per_video))])
          if cfg.NET.USE_TEMPORAL_ATT:
            with tf.variable_scope('TemporalAttention'):
              logits = tf.expand_dims(logits, axis=-2)  #[bs, 3, 1, nc]
              logits_att = slim.conv2d(
                logits, 1, [1, 1],
                weights_initializer=random_normal(0.001),
                biases_initializer=tf.constant_initializer(
                  1.0 / logits.get_shape().as_list()[1]),
                activation_fn=None, normalizer_fn=None)
              logits = logits * logits_att
              logits = tf.squeeze(logits, axis=-2)
              end_points['TemporalAttention'] = logits_att
          logits = tf.reduce_mean(logits, axis=1)
      return logits, end_points

  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
