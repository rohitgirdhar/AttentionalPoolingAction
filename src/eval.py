"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import sys
import tensorflow as tf
import pprint
import os
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pdb

from datasets import dataset_factory
sys.path.append('../models/slim')
from nets import nets_factory
from preprocessing import preprocessing_factory
from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from eval.utils import compute_map
from preprocess_pipeline import get_input

slim = tf.contrib.slim

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--gpu', dest='gpu',
                      help='GPU to use for running this.',
                      default='0', type=str)
  parser.add_argument('--save', dest='save', action='store_const',
                      const=True, default=False,
                      help='Set to save the features. Works only in mAP mode. '
                           '(Set in cfg).')
  parser.add_argument('--outfpath', default=None,
                      help='(Optional) Give a custom path to save the features. '
                           'By def. picks a path in ckpt directory.')
  parser.add_argument('--preprocs', default=[], nargs='*',
                      help='Set additional preprocs to do when testing. Eg. '
                           'can put \'flips\'. This will flip images before '
                           'pushing through the network. Can be useful for '
                           'late fusion of multiple features.')
  parser.add_argument('--ept', dest='ept', nargs='+', type=str, default=[],
                      help='Optional end point to store. '
                           'By def store the softmax logits.')
  parser.add_argument('--split_name', default=None, type=str,
                      help='Set to change the dataset split to run on. '
                           'Eg, \'train\' or \'test\'.')
  parser.add_argument('--frames_per_video', default=None, type=int,
                      help='Set to change the '
                           'cfg.TRAIN.VIDEO_FRAMES_PER_VIDEO.')
  parser.add_argument('--dataset_list_dir', default=None, type=str,
                      help='Set to change the train_test_lists dir.')
  args = parser.parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

  # Change config for some options
  if args.split_name is not None:
    cfg.TEST.DATASET_SPLIT_NAME = args.split_name
  if args.frames_per_video is not None:
    cfg.TEST.VIDEO_FRAMES_PER_VIDEO = args.frames_per_video
  if args.outfpath is not None:
    args.save = True
  return args, cfg


def mkdir_p(dpath):
  try:
    os.makedirs(dpath)
  except:
    pass


def main():
  args, cfg = parse_args()
  train_dir = get_output_dir('default' if args.cfg_file is None
                             else args.cfg_file)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  print('Using Config:')
  pprint.pprint(cfg)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    kwargs = {}
    if cfg.TEST.VIDEO_FRAMES_PER_VIDEO > 1:
      kwargs['num_samples'] = cfg.TEST.VIDEO_FRAMES_PER_VIDEO
      kwargs['modality'] = cfg.INPUT.VIDEO.MODALITY
      kwargs['split_id'] = cfg.INPUT.SPLIT_ID
    if args.dataset_list_dir is not None:
      kwargs['dataset_list_dir'] = args.dataset_list_dir
    elif cfg.DATASET_LIST_DIR != '':
      kwargs['dataset_list_dir'] = cfg.DATASET_LIST_DIR
    if cfg.INPUT_FILE_STYLE_LABEL != '':
      kwargs['input_file_style_label'] = cfg.INPUT_FILE_STYLE_LABEL
    dataset, num_pose_keypoints = dataset_factory.get_dataset(
        cfg.DATASET_NAME, cfg.TEST.DATASET_SPLIT_NAME, cfg.DATASET_DIR,
        **kwargs)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        cfg.MODEL_NAME,
        num_classes=dataset.num_classes,
        num_pose_keypoints=num_pose_keypoints,
        is_training=False,
        cfg=cfg)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        num_epochs=1,
        common_queue_capacity=2 * cfg.TEST.BATCH_SIZE,
        common_queue_min=cfg.TEST.BATCH_SIZE)
    [image, action_label] = get_input(provider, cfg,
                                      ['image', 'action_label'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = cfg.MODEL_NAME
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = cfg.TRAIN.IMAGE_SIZE or network_fn.default_image_size

    image = image_preprocessing_fn(
      image, eval_image_size, eval_image_size,
      resize_side_min=cfg.TRAIN.RESIZE_SIDE,
      resize_side_max=cfg.TRAIN.RESIZE_SIDE)

    # additional preprocessing as required
    if 'flips' in args.preprocs:
      tf.logging.info('Flipping all images while testing!')
      image = tf.stack([
        tf.image.flip_left_right(el) for el in tf.unstack(image)])

    images, action_labels = tf.train.batch(
      [image, action_label],
      batch_size=cfg.TEST.BATCH_SIZE,
      # following is because if there are more, the order of batch can be
      # different due to different speed... so avoid that
      # http://stackoverflow.com/questions/35001027/does-batching-queue-tf-train-batch-not-preserve-order#comment57731040_35001027
      # num_threads=1 if args.save else cfg.NUM_PREPROCESSING_THREADS,
      num_threads=1,  # The above was too unsafe as sometimes I forgot --save
                      # and it would just randomize the whole thing.
                      # This is very important so
                      # shifting to this by default. Better safe than sorry.
      allow_smaller_final_batch=True if cfg.TEST.VIDEO_FRAMES_PER_VIDEO == 1
                                else False,  # because otherwise we need to
                                             # average logits over the frames,
                                             # and that needs first dimensions
                                             # to be fully defined
      capacity=5 * cfg.TEST.BATCH_SIZE)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)
    end_points['images'] = images

    if cfg.TEST.MOVING_AVERAGE_DECAY:
      variable_averages = tf.train.ExponentialMovingAverage(
          cfg.TEST.MOVING_AVERAGE_DECAY, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    if cfg.TRAIN.LOSS_FN_ACTION.startswith('multi-label'):
      logits = tf.sigmoid(logits)
    else:
      logits = tf.nn.softmax(logits, -1)
    labels = tf.squeeze(action_labels)
    end_points['labels'] = labels

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        # 'Recall@5': slim.metrics.streaming_recall_at_k(
        #     logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if cfg.TEST.MAX_NUM_BATCHES:
      num_batches = cfg.TEST.MAX_NUM_BATCHES
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(cfg.TEST.BATCH_SIZE))

    # just test the latest trained model
    checkpoint_path = cfg.TEST.CHECKPOINT_PATH or train_dir
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint_path = checkpoint_path
    checkpoint_step = int(checkpoint_path.split('-')[-1])

    tf.logging.info('Evaluating %s' % checkpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    summary_writer = tf.summary.FileWriter(logdir=train_dir)

    if cfg.TEST.EVAL_METRIC == 'mAP' or args.save or args.ept:
      from tensorflow.python.training import supervisor
      from tensorflow.python.framework import ops
      import h5py
      saver = tf.train.Saver(variables_to_restore)
      sv = supervisor.Supervisor(graph=ops.get_default_graph(),
                                 logdir=None,
                                 summary_op=None,
                                 summary_writer=summary_writer,
                                 global_step=None,
                                 saver=None)
      all_labels = []
      end_points['logits'] = logits
      end_points_to_save = args.ept + ['logits']
      end_points_to_save = list(set(end_points_to_save))
      all_feats = dict([(ename, []) for ename in end_points_to_save])
      with sv.managed_session(
          '', start_standard_services=False,
          config=config) as sess:
        saver.restore(sess, checkpoint_path)
        sv.start_queue_runners(sess)
        for j in tqdm(range(int(math.ceil(num_batches)))):
          feats = sess.run([
            action_labels,
            [end_points[ename] for ename in end_points_to_save]])
          all_labels.append(feats[0])
          for ept_id, ename in enumerate(end_points_to_save):
            all_feats[ename].append(feats[1][ept_id])
      APs = []
      all_labels = np.concatenate(all_labels)
      if args.save or args.ept:
        res_outdir = os.path.join(train_dir, 'Features/')
        mkdir_p(res_outdir)
        outfpath = args.outfpath or os.path.join(
          res_outdir, 'features_ckpt_{}_{}.h5'.format(
          cfg.TEST.DATASET_SPLIT_NAME,
          checkpoint_step))
        print('Saving the features/logits/labels to {}'.format(outfpath))
        with h5py.File(outfpath, 'a') as fout:
          for ename in end_points_to_save:
            if ename in fout:
              tf.logging.warning('Deleting {} from output HDF5 to write the '
                                 'new features.'.format(ename))
              del fout[ename]
            if ename == 'labels':
              feat_to_save = np.array(all_feats[ename])
            else:
              feat_to_save = np.concatenate(all_feats[ename])
            try:
              fout.create_dataset(
                ename, data=feat_to_save,
                compression='gzip', compression_opts=9)
            except:
              pdb.set_trace()  # manually deal with it and continue
          if 'labels' in fout:
            del fout['labels']
          fout.create_dataset(
            'labels', data=all_labels,
            compression='gzip', compression_opts=9)

      if args.ept:
        tf.logging.info('Evaluation had --ept passed in. '
                        'This indicates script was used for feature '
                        'extraction. Hence, not performing any evaluation.')
        return
      # Evaluation code
      all_logits = np.concatenate(all_feats['logits'])
      acc = np.mean(
        all_logits.argmax(axis=1) == all_labels)
      mAP = compute_map(all_logits, all_labels)[0]
      print('Mean AP: {}'.format(mAP))
      print('Accuracy: {}'.format(acc))
      summary_writer.add_summary(tf.Summary(value=[
        tf.Summary.Value(
          tag='mAP/{}'.format(cfg.TEST.DATASET_SPLIT_NAME),
          simple_value=mAP)]),
        global_step=checkpoint_step)
      summary_writer.add_summary(tf.Summary(value=[
        tf.Summary.Value(
          tag='Accuracy/{}'.format(cfg.TEST.DATASET_SPLIT_NAME),
          simple_value=acc)]),
        global_step=checkpoint_step)
    else:
      slim.evaluation.evaluate_once(
        master='',
        checkpoint_path=checkpoint_path,
        logdir=train_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        variables_to_restore=variables_to_restore,
        session_config=config)


if __name__ == '__main__':
  main()
