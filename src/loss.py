import tensorflow as tf
slim = tf.contrib.slim

def gen_losses(
  labels_action, logits_action, loss_type_action, num_action_classes,
  action_loss_wt,
  labels_pose, logits_pose, loss_type_pose, labels_pose_valid, pose_loss_wt,
  end_points, cfg):

  with tf.name_scope('LossFn'):
    if loss_type_pose and logits_pose.get_shape().as_list()[-1] > 0:
      with tf.name_scope('PoseLoss'):
        # Loss over the pose
        if labels_pose.get_shape().as_list() != \
           logits_pose.get_shape().as_list():
          tf.logging.info('Sizes of logits {} and labels {} are different. '
                          'Change the cfg.FINAL_POSE_HMAP_SIDE to avoid '
                          'a resize operation.'.format(
                            logits_pose.get_shape().as_list(),
                            labels_pose.get_shape().as_list()))
          labels_pose = tf.image.resize_images(
            labels_pose, logits_pose.get_shape().as_list()[-3:-1])
        # ignore the unknown channels, set those channels to 0 to incur no loss

        # Following needs defining the gradient for this...
        # labels_pose = zero_out_channels(labels_pose, labels_pose_valid)
        # logits_pose = zero_out_channels(logits_pose, labels_pose_valid)

        with tf.name_scope('ValidPoseLoss'):
          channels_valid = tf.unstack(labels_pose_valid, axis=-1)
          channels_logits = tf.unstack(logits_pose, axis=-1)
          channels_labels = tf.unstack(labels_pose, axis=-1)
          loss_elements = []
          pose_loss_mask = []
          for v, lbl, lgt in zip(channels_valid, channels_logits, channels_labels):
            if cfg.TRAIN.LOSS_FN_POSE_SAMPLED:
              # To make it harder
              neg_areas = tf.equal(lgt, 0)
              pos_areas = tf.greater(lgt, 0)
              total_area = lgt.shape.num_elements()
              pos_area_ratio = tf.reduce_sum(tf.to_float(pos_areas)) / total_area
              # select that much of neg area
              neg_areas_selected = tf.to_float(tf.less(tf.random_uniform(
                tf.shape(lgt), 0, 1.0),
                pos_area_ratio)) * tf.to_float(neg_areas)
              # keep all positive pixels
              mask = tf.greater(neg_areas_selected + tf.to_float(
                tf.greater(lbl, 0)), 0)
              mask = tf.to_float(mask)
              lgt = lgt * mask  # just do loss over this subset
              lbl = lbl * mask
              loss_val = 0.5 * tf.reduce_mean(tf.square(lbl - lgt), axis=[1,2])
            else:
              mask = tf.ones(tf.shape(lgt))
              loss_val = 0.5 * tf.reduce_sum(
                tf.square(lbl - lgt), axis=[1,2]) / tf.reduce_sum(mask)
            pose_loss_mask.append(tf.expand_dims(mask, -1))
            if loss_type_pose == 'l2':
              L = tf.reduce_mean(tf.where(
                v,
                loss_val,
                [0] * v.get_shape().as_list()[0]))
            elif loss_type_pose == '':
              L = 0
            else:
              raise ValueError('Invalid loss {}'.format(loss_type_pose))
            loss_elements.append(L)
        end_points['PoseLossMask'] = tf.concat(pose_loss_mask, axis=-1)
        tot_loss = tf.reduce_sum(loss_elements, name='ValidPoseEucLoss')
        tf.losses.add_loss(tot_loss * pose_loss_wt)

    with tf.name_scope('ActionLoss'):
      # TODO (rgirdhar): Add the option of having -1 label, so ignore that one
      if loss_type_action == 'softmax-xentropy':
        tf.losses.softmax_cross_entropy(
          onehot_labels=slim.one_hot_encoding(
            labels_action,
            num_action_classes),
          logits=logits_action,
          weights=action_loss_wt)
      elif loss_type_action == 'l2':
        tf.losses.mean_squared_error(
          labels=slim.one_hot_encoding(
            labels_action,
            num_action_classes),
          predictions=logits_action,
          weights=action_loss_wt)
      elif loss_type_action == 'multi-label':
        labels_action = tf.to_float(labels_action)
        # labels_action = tf.Print(
        #   labels_action, [labels_action, tf.reduce_sum(labels_action, 1)],
        #   "Label action:")
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
          targets=labels_action,
          logits=logits_action,
          pos_weight=10))
        tf.losses.add_loss(loss)
      elif loss_type_action == 'multi-label-2':
        tf.losses.sigmoid_cross_entropy(
          multi_class_labels=labels_action,
          logits=logits_action)
      elif loss_type_action == '':
        tf.logging.info('No loss on action')
      else:
        raise ValueError('Unrecognized loss {}'.format(loss_type_action))
