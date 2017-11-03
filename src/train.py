from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sys
import pprint
import os
import time
import numpy as np
from datetime import datetime

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import timeline
from datasets import dataset_factory
sys.path.append('../models/slim')
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from restore import model_restorer
from loss import gen_losses
from preprocess_pipeline import train_preprocess_pipeline

slim = tf.contrib.slim

def _configure_learning_rate(num_samples_per_epoch, num_clones, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  if cfg.TRAIN.NUM_STEPS_PER_DECAY > 0:
    decay_steps = cfg.TRAIN.NUM_STEPS_PER_DECAY
    tf.logging.info('Using {} steps for decay. Ignoring any epoch setting for '
                    'decay.'.format(decay_steps))
  else:
    decay_steps = int(num_samples_per_epoch / (
      cfg.TRAIN.BATCH_SIZE * num_clones * cfg.TRAIN.ITER_SIZE) * cfg.TRAIN.NUM_EPOCHS_PER_DECAY)

  if cfg.TRAIN.LEARNING_RATE_DECAY_TYPE == 'exponential':
    return tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      cfg.TRAIN.LEARNING_RATE_DECAY_RATE,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif cfg.TRAIN.LEARNING_RATE_DECAY_TYPE == 'fixed':
    return tf.constant(cfg.TRAIN.LEARNING_RATE, name='fixed_learning_rate')
  elif cfg.TRAIN.LEARNING_RATE_DECAY_TYPE == 'polynomial':
    return tf.train.polynomial_decay(cfg.TRAIN.LEARNING_RATE,
                                     global_step,
                                     decay_steps,
                                     cfg.TRAIN.END_LEARNING_RATE,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     cfg.TRAIN.LEARNING_RATE_DECAY_RATE)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if cfg.optimizer is not recognized.
  """
  if cfg.TRAIN.OPTIMIZER == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=cfg.TRAIN.ADAM_BETA1,
        beta2=cfg.TRAIN.ADAM_BETA2,
        epsilon=cfg.TRAIN.OPT_EPSILON)
  elif cfg.TRAIN.OPTIMIZER == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=cfg.TRAIN.MOMENTUM,
        name='Momentum')
  elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=cfg.TRAIN.RMSPROP_DECAY,
        momentum=cfg.TRAIN.MOMENTUM,
        epsilon=cfg.TRAIN.OPT_EPSILON)
  elif cfg.TRAIN.OPTIMIZER == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', cfg.TRAIN.OPTIMIZER)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.histogram_summary(variable.op.name, variable))
  summaries.append(tf.summary.scalar(tensor=learning_rate,
                                     name='training/Learning Rate'))
  return summaries


def _get_init_fn(train_dir):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if cfg.TRAIN.CHECKPOINT_PATH is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % train_dir)
    return None

  exclusions = []
  if cfg.TRAIN.CHECKPOINT_EXCLUDE_SCOPES:
    exclusions = [scope.strip()
                  for scope in cfg.TRAIN.CHECKPOINT_EXCLUDE_SCOPES.split(',')]

  # variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)
  # NOTE: The above was wrong!! It would restore all global_step, momentum etc
  # variables too, which we don't want when starting from a pretrained model
  # (like imagenet). The above is (and should be) used when restoring from a
  # half-trained model of the same script (which doesn't happen here anyway,
  # see above, there's a return None if a checkpoint exists)
  variables_to_restore = slim.filter_variables(
    slim.get_model_variables(),
    exclude_patterns=exclusions)

  if tf.gfile.IsDirectory(cfg.TRAIN.CHECKPOINT_PATH):
    checkpoint_path = tf.train.latest_checkpoint(cfg.TRAIN.CHECKPOINT_PATH)
  else:
    checkpoint_path = cfg.TRAIN.CHECKPOINT_PATH

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return model_restorer.restore_model(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=cfg.TRAIN.IGNORE_MISSING_VARS,
      var_name_mapper_type=cfg.TRAIN.VAR_NAME_MAPPER)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if cfg.TRAIN.TRAINABLE_SCOPES == '':
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in cfg.TRAIN.TRAINABLE_SCOPES.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def _gen_overlayed_img(hmap, img):
  with tf.name_scope('VisualizeOverlayedHeatmap'):
    hmap = tf.expand_dims(hmap, -1)
    hmap = tf.image.resize_images(
      hmap, img.get_shape().as_list()[-3:-1])
    img = tf.tile(
      tf.image.rgb_to_grayscale(img), [1, 1, 1, 3])
    hmap = tf.image.grayscale_to_rgb(hmap)
    hmap = tf.concat([
      tf.expand_dims(hmap[..., 0] * 255.0, -1),
      hmap[..., 1:] * 0.0], axis=-1)
    return (0.5 * img + 0.5 * hmap)


def _summarize_heatmaps(name, tensor, img_tensor):
  # return tf.summary.image(name, tf.reduce_sum(tensor, axis=-1, keep_dims=True))
  if tensor.get_shape()[-1] == 0:
    tf.logging.info('Pose heatmaps have 0 channels. Not summarizing')
    return set()
  return set([
    tf.summary.image(
      name + '/head', _gen_overlayed_img(tensor[..., 9], img_tensor)),
    tf.summary.image(
      name + '/lwrist', _gen_overlayed_img(tensor[..., 15], img_tensor)),
    tf.summary.image(
      name + '/rankle', _gen_overlayed_img(tensor[..., 0], img_tensor)),
    tf.summary.image(
      name + '/pelvis', _gen_overlayed_img(tensor[..., 6], img_tensor))])


end_points_debug = {}
def _train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  if cfg.TRAIN.ITER_SIZE == 1:
    # To Debug, uncomment here and observe the end_points_debug
    total_loss, np_global_step = sess.run([train_op, global_step],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)
  else:
    for j in range(cfg.TRAIN.ITER_SIZE-1):
      sess.run([train_op[j]])
    total_loss, np_global_step = sess.run([
      train_op[cfg.TRAIN.ITER_SIZE-1], global_step],
      options=trace_run_options,
      run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    tf.logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      tf.logging.info('%s: global step %d: loss = %.4f (%.2f sec/step)',
                   datetime.now(), np_global_step, total_loss, time_elapsed)

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

  tf.logging.info('Using Config:')
  pprint.pprint(cfg)

  train_dir = get_output_dir('default' if args.cfg_file is None
                             else args.cfg_file)
  os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPUS
  num_clones = len(cfg.GPUS.split(','))

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    ######################
    # Config model_deploy#
    ######################
    tf.set_random_seed(cfg.RNG_SEED)
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=False,
        replica_id=0,
        num_replicas=1,
        num_ps_tasks=0)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    kwargs = {}
    if cfg.TRAIN.VIDEO_FRAMES_PER_VIDEO > 1:
      kwargs['num_samples'] = cfg.TRAIN.VIDEO_FRAMES_PER_VIDEO
      kwargs['randomFromSegmentStyle'] = cfg.TRAIN.READ_SEGMENT_STYLE
      kwargs['modality'] = cfg.INPUT.VIDEO.MODALITY
      kwargs['split_id'] = cfg.INPUT.SPLIT_ID
    if cfg.DATASET_LIST_DIR != '':
      kwargs['dataset_list_dir'] = cfg.DATASET_LIST_DIR
    if cfg.INPUT_FILE_STYLE_LABEL != '':
      kwargs['input_file_style_label'] = cfg.INPUT_FILE_STYLE_LABEL
    dataset, num_pose_keypoints = dataset_factory.get_dataset(
      cfg.DATASET_NAME, cfg.TRAIN.DATASET_SPLIT_NAME, cfg.DATASET_DIR,
      **kwargs)

    ####################
    # Select the network #
    ####################
    network_fn = nets_factory.get_network_fn(
        cfg.MODEL_NAME,
        num_classes=(dataset.num_classes),
        num_pose_keypoints=num_pose_keypoints,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        is_training=True,
        cfg=cfg)  # advanced network creation controlled with cfg.NET

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = cfg.MODEL_NAME
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=cfg.NUM_READERS,
          common_queue_capacity=20 * cfg.TRAIN.BATCH_SIZE,
          common_queue_min=10 * cfg.TRAIN.BATCH_SIZE)

      [image, pose_label_hmap,
       pose_label_valid, action_label] = train_preprocess_pipeline(
         provider, cfg, network_fn, num_pose_keypoints,
         image_preprocessing_fn)
      # input_data = [preprocess_pipeline(
      #   provider, cfg, network_fn, num_pose_keypoints, image_preprocessing_fn)
      #   for _ in range(cfg.NUM_PREPROCESSING_THREADS)]

      images, pose_labels_hmap, pose_labels_valid, action_labels = tf.train.batch(
          [image, pose_label_hmap, pose_label_valid, action_label],
          # input_data,
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_threads=cfg.NUM_PREPROCESSING_THREADS,
          capacity=5 * cfg.TRAIN.BATCH_SIZE)
      batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, pose_labels_hmap, pose_labels_valid, action_labels],
        capacity=5 * deploy_config.num_clones * cfg.TRAIN.ITER_SIZE)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels_pose, labels_pose_valid, labels_action = batch_queue.dequeue()
      # due to the multi-frame/video thing, need to squeeze first 2 dimensions
      labels_pose = tf.concat(tf.unstack(labels_pose), axis=0)
      labels_pose_valid = tf.concat(tf.unstack(labels_pose_valid), axis=0)
      logits, end_points = network_fn(images)
      pose_logits = end_points['PoseLogits']

      #############################
      # Specify the loss function #
      #############################
      # if 'AuxLogits' in end_points:
      #   slim.losses.softmax_cross_entropy(
      #       end_points['AuxLogits'], labels,
      #       label_smoothing=cfg.TRAIN.LABEL_SMOOTHING, weight=0.4, scope='aux_loss')
      # slim.losses.softmax_cross_entropy(
      #     logits, labels, label_smoothing=cfg.TRAIN.LABEL_SMOOTHING, weight=1.0)
      end_points['Images'] = images
      end_points['PoseLabels'] = labels_pose
      end_points['ActionLabels'] = labels_action
      end_points['ActionLogits'] = logits
      tf.logging.info('PoseLogits shape is {}.'.format(pose_logits.get_shape().as_list()))

      gen_losses(labels_action, logits, cfg.TRAIN.LOSS_FN_ACTION,
                 dataset.num_classes, cfg.TRAIN.LOSS_FN_ACTION_WT,
                 labels_pose, pose_logits, cfg.TRAIN.LOSS_FN_POSE,
                 labels_pose_valid, cfg.TRAIN.LOSS_FN_POSE_WT, end_points, cfg)

      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs

    # store the end points in a global variable for debugging in train_step
    global end_points_debug
    end_points_debug = end_points

    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      # summaries.add(tf.summary.scalar(tf.nn.zero_fraction(x),
      #                                 name='sparsity/' + end_point))
    sum_img = tf.concat(tf.unstack(end_points['Images']), axis=0)
    if sum_img.get_shape().as_list()[-1] not in [1, 3, 4]:
      sum_img = tf.reduce_sum(sum_img, axis=-1, keep_dims=True)
      sum_img = sum_img - tf.reduce_min(sum_img)
      sum_img = sum_img / (tf.reduce_max(sum_img) + cfg.EPS)
    summaries.add(tf.summary.image('images', sum_img))
    for epname in cfg.TRAIN.OTHER_IMG_SUMMARIES_TO_ADD:
      if epname in end_points:
        summaries.add(tf.summary.image('image_vis/' + epname, end_points[epname]))
    summaries = summaries.union(_summarize_heatmaps(
      'labels', end_points['PoseLabels'], sum_img))
    summaries = summaries.union(_summarize_heatmaps(
      'pose', end_points['PoseLogits'], sum_img))
    if 'PoseLossMask' in end_points:
      summaries = summaries.union(_summarize_heatmaps(
        'loss_mask/pose', end_points['PoseLossMask'], sum_img))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar(tensor=loss, name='losses/%s' % loss.op.name))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if cfg.TRAIN.MOVING_AVERAGE_VARIABLES:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          cfg.TRAIN.MOVING_AVERAGE_VARIABLES, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, num_clones, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar(tensor=learning_rate,
                                      name='learning_rate'))

    # if cfg.sync_replicas:
    #   # If sync_replicas is enabled, the averaging will be done in the chief
    #   # queue runner.
    #   optimizer = tf.train.SyncReplicasOptimizer(
    #       opt=optimizer,
    #       replicas_to_aggregate=,
    #       variable_averages=variable_averages,
    #       variables_to_average=moving_average_variables,
    #       replica_id=tf.constant(cfg.task, tf.int32, shape=()),
    #       total_num_replicas=cfg.worker_replicas)
    # elif cfg.moving_average_decay:
    #   # Update ops executed locally by trainer.
    #   update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()
    tf.logging.info('Training the following variables: {}'.format(
                    ', '.join([var.op.name for var in variables_to_train])))

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train,
        clip_gradients=cfg.TRAIN.CLIP_GRADIENTS)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar(tensor=total_loss,
                                    name='total_loss'))

    # Create gradient updates.
    train_ops = {}
    if cfg.TRAIN.ITER_SIZE == 1:
      grad_updates = optimizer.apply_gradients(clones_gradients,
                                               global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                        name='train_op')
      train_ops = train_tensor
    else:
      with tf.name_scope('AccumulateGradients'):
        # copied as is from my previous code
        gvs = [(grad, var) for grad, var in clones_gradients]
        varnames = [var.name for grad, var in gvs]
        varname_to_var = {var.name: var for grad, var in gvs}
        varname_to_grad = {var.name: grad for grad, var in gvs}
        varname_to_ref_grad = {}
        for vn in varnames:
          grad = varname_to_grad[vn]
          print("accumulating ... ", (vn, grad.get_shape()))
          with tf.variable_scope("ref_grad"):
            with tf.device(deploy_config.variables_device()):
              ref_var = slim.local_variable(
                  np.zeros(grad.get_shape(),dtype=np.float32),
                  name=vn[:-2])
              varname_to_ref_grad[vn] = ref_var

        all_assign_ref_op = [ref.assign(varname_to_grad[vn]) for vn, ref in varname_to_ref_grad.items()]
        all_assign_add_ref_op = [ref.assign_add(varname_to_grad[vn]) for vn, ref in varname_to_ref_grad.items()]
        assign_gradients_ref_op = tf.group(*all_assign_ref_op)
        accmulate_gradients_op = tf.group(*all_assign_add_ref_op)
        with tf.control_dependencies([accmulate_gradients_op]):
          final_gvs = [(varname_to_ref_grad[var.name] / float(cfg.TRAIN.ITER_SIZE), var) for grad, var in gvs]
          apply_gradients_op = optimizer.apply_gradients(final_gvs, global_step=global_step)
          update_ops.append(apply_gradients_op)
          update_op = tf.group(*update_ops)
          train_tensor = control_flow_ops.with_dependencies([update_op],
              total_loss, name='train_op')
        for i in range(cfg.TRAIN.ITER_SIZE):
          if i == 0:
            train_ops[i] = assign_gradients_ref_op
          elif i < cfg.TRAIN.ITER_SIZE - 1:  # because apply_gradients also computes
                                             # (see control_dependency), so
                                             # no need of running an extra iteration
            train_ops[i] = accmulate_gradients_op
          else:
            train_ops[i] = train_tensor

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 4  # to avoid too many threads
    # The following seems optimal... though not sure
    config.inter_op_parallelism_threads = max(
      cfg.NUM_PREPROCESSING_THREADS, 12)
    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_ops,
        train_step_fn=_train_step,
        logdir=train_dir,
        master='',
        is_chief=True,
        init_fn=_get_init_fn(train_dir),
        summary_op=summary_op,
        number_of_steps=cfg.TRAIN.MAX_NUMBER_OF_STEPS,
        log_every_n_steps=cfg.TRAIN.LOG_EVERY_N_STEPS,
        save_summaries_secs=cfg.TRAIN.SAVE_SUMMARIES_SECS,
        save_interval_secs=cfg.TRAIN.SAVE_INTERVAL_SECS,
        sync_optimizer=None,
        session_config=config)


if __name__ == '__main__':
  main()
