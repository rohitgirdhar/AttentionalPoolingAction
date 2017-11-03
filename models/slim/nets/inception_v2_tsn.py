"""Contains the definition for inception v2 (TSN) classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
random_normal = lambda stddev: tf.random_normal_initializer(0.0, stddev)

def conv_set(net, num_outputs, filter_size, stride=1, weight_std=0.001,
             padding=0):
  if padding > 0:
    net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
  net = slim.conv2d(
    net, num_outputs, filter_size,
    stride=stride,
    padding='VALID')
  net = slim.batch_norm(net,
                        updates_collections=tf.GraphKeys.UPDATE_OPS,
                        epsilon=1e-5,
                        decay=0.9,
                        scale=True)
  net = tf.nn.relu(net)
  return net


def pool(net, pool_type='avg', kernel=3, stride=1, padding=0):
  if pool_type == 'avg':
    fn = slim.avg_pool2d
  elif pool_type == 'max':
    fn = slim.max_pool2d
  else:
    raise ValueError('Unknown pool type')
  with tf.name_scope('%s_pool' % pool_type):
    net = fn(net, [kernel, kernel], stride=stride,
             padding='VALID' if padding==0 else 'SAME')
  return net


def inception_module(net, small_module=False,
                     num_outputs=[64,64,64,32,64,96,96],
                     force_max_pool=False):
  all_nets = []
  if not small_module:
    with tf.variable_scope('1x1'):
      net_1 = conv_set(net, num_outputs[0], [1, 1])
    all_nets.append(net_1)

  with tf.variable_scope('3x3_reduce'):
    net_2 = conv_set(net, num_outputs[1], [1, 1])
  with tf.variable_scope('3x3'):
    net_2 = conv_set(net_2, num_outputs[2], [3, 3],
                     padding=1,
                     stride=2 if small_module else 1)
  all_nets.append(net_2)

  with tf.variable_scope('double_3x3_reduce'):
    net_3 = conv_set(net, num_outputs[4], [1, 1])
  with tf.variable_scope('double_3x3_1'):
    net_3 = conv_set(net_3, num_outputs[5], [3, 3], padding=1)
  with tf.variable_scope('double_3x3_2'):
    net_3 = conv_set(net_3, num_outputs[6], [3, 3], padding=1,
                     stride=2 if small_module else 1)
  all_nets.append(net_3)

  with tf.variable_scope('pool'):
    if small_module:
      net_4 = pool(net, 'max', 3, 2, 1)
    elif force_max_pool:
      net_4 = pool(net, 'max', 3, 1, 1)
    else:
      net_4 = pool(net, 'avg', 3, 1, 1)
  if not small_module:
    with tf.variable_scope('pool_proj'):
      net_4 = conv_set(net_4, num_outputs[3], [1, 1])
  all_nets.append(net_4)

  net = tf.concat(all_nets, 3)
  return net


def inception_v2_tsn_base(inputs,
                          final_endpoint='Mixed_5c',
                          min_depth=16,
                          depth_multiplier=1.0,
                          scope=None,
                          is_training=False,
                          train_top_bn=False):
  """Inception v2 (TSN code).

  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  with tf.variable_scope(scope, 'InceptionV2_TSN', [inputs]):
      # 224 x 224 x 3
      end_point = 'conv1/7x7_s2'
      with tf.variable_scope(end_point):
        with slim.arg_scope(
          [slim.batch_norm],
          is_training=is_training if train_top_bn else False,
          trainable=True if train_top_bn else False):
          net = conv_set(inputs, 64, [7, 7],
                         stride=2,
                         padding=3)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 112 x 112 x 64
      end_point = 'pool1/3x3_s2'
      net = slim.max_pool2d(net, [3, 3], scope=end_point,
                            stride=2, padding='SAME')
      # net = pool(net, 'max', 3, 2, 1)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 56 x 56 x 64
      end_point = 'conv2/3x3_reduce'
      with tf.variable_scope(end_point):
        net = conv_set(net, 64, [1, 1], weight_std=0.1,
                       padding=0)
      # net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2,
      #                       padding='SAME')
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points
      end_point = 'conv2/3x3'
      with tf.variable_scope(end_point):
        net = conv_set(net, 192, [3, 3], weight_std=0.1, padding=1)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points
      end_point = 'pool2/3x3_s2'
      net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2,
                            padding='SAME')
      # net = pool(net, 'max', 3, 2, 1)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      # Inception module.
      end_point = 'inception_3a'
      with tf.variable_scope(end_point):
        net = inception_module(net)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_3b'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[64,64,96,64,64,96,96])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_3c'
      with tf.variable_scope(end_point):
        net = inception_module(net, small_module=True,
                               num_outputs=[-1,128,160,-1,64,96,96])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_4a'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[224,64,96,128,96,128,128])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_4b'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[192,96,128,128,96,128,128])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_4c'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[160,128,160,128,128,160,160])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_4d'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[96,128,192,128,160,192,192])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_4e'
      with tf.variable_scope(end_point):
        net = inception_module(net, small_module=True,
                               num_outputs=[-1,128,192,-1,192,256,256])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_5a'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[352,192,320,128,160,224,224])
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

      end_point = 'inception_5b'
      with tf.variable_scope(end_point):
        net = inception_module(net, num_outputs=[352,192,320,128,192,224,224],
                              force_max_pool=True)
      end_points[tf.get_variable_scope().name + '/' + end_point] = net
      if end_point == final_endpoint: return net, end_points

  return net, end_points


def inception_v2_tsn(inputs,
                     num_classes=1000,
                     is_training=True,
                     dropout_keep_prob=0.2,
                     min_depth=16,
                     depth_multiplier=1.0,
                     prediction_fn=slim.softmax,
                     spatial_squeeze=True,
                     reuse=None,
                     conv_only=None,
                     conv_endpoint='inception_5b',
                     # conv_endpoint='inception_5a',  # testing for now
                     train_top_bn=False,
                     scope='InceptionV2_TSN'):
  """Inception v2 model for video classification.

  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV2_TSN', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.dropout],
                        is_training=is_training):
      with slim.arg_scope([slim.batch_norm],
                          is_training=False,
                          trainable=False):
        net, end_points = inception_v2_tsn_base(
            inputs, scope=scope, min_depth=min_depth,
            depth_multiplier=depth_multiplier,
            final_endpoint=conv_endpoint if conv_only else None,
            is_training=is_training,
            train_top_bn=train_top_bn)
        if conv_only:
          return net, end_points
        with tf.variable_scope('Logits'):
          kernel_size = _reduced_kernel_size_for_small_input(net, [100, 100])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID', stride=1,
                                scope='AvgPool_Logits_{}x{}'.format(*kernel_size))
          # The following would give the same output/performance too.
          # net = tf.reduce_mean(net, axis=[1,2], keep_dims=True)
          # 1 x 1 x 1024
          logging.info('Using dropout %f' % (1-dropout_keep_prob))
          net = slim.dropout(net, keep_prob=dropout_keep_prob,
                             scope='Dropout_Logits')
          logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                               normalizer_fn=None,
                               weights_initializer=random_normal(0.001),
                               biases_initializer=init_ops.zeros_initializer())
          if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v2_tsn.default_image_size = 224


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.pack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def inception_v2_tsn_arg_scope(weight_decay=0.00004):
  """Defines the default InceptionV2 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # Allow a gamma variable
      'scale': True,
  }

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,  # manually added later, as I need to add BN after
                             # the convolution
        biases_initializer=init_ops.constant_initializer(value=0.2),
        normalizer_fn=None) as sc:
      return sc
