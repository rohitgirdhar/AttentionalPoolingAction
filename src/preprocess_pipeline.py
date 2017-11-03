import tensorflow as tf
from custom_ops.custom_ops_factory import pose_to_heatmap, render_pose, \
    render_objects, extract_glimpse

def _resize_if_needed(image, max_wd):
  with tf.name_scope('LimitMaxSizeOriginalImage'):
    im_ht = tf.shape(image)[-3]
    im_wd = tf.shape(image)[-2]
    new_ht = tf.cast(im_ht, tf.float32) * (
      tf.cast(max_wd, tf.float32) / tf.cast(im_wd, tf.float32))
    new_ht = tf.cast(new_ht, tf.int64)
    image = tf.cond(
      tf.greater(im_wd, max_wd),
      lambda: tf.image.resize_images(
        image, tf.cast([new_ht, max_wd], tf.int32)),
      lambda: tf.cast(image, tf.float32))
    image = tf.cast(image, tf.uint8)
  return image


def _replay_augmentation(H, aug_info):
  # use the augmentation info from the original image to identically transform
  # the heatmap H
  with tf.name_scope('ReplayAugmentation'):
    ## 1. Crop
    H_wd = tf.shape(H)[-2]
    H_ht = tf.shape(H)[-3]
    num_channels = tf.shape(H)[-1]
    orig_wd = aug_info['image_shape'][-2]
    orig_ht = aug_info['image_shape'][-3]
    ratio_x = tf.to_float(H_wd) / tf.to_float(orig_wd)
    ratio_y = tf.to_float(H_ht) / tf.to_float(orig_ht)
    start_points = [tf.to_float(aug_info['crop_info'][0]) * ratio_y,
                    tf.to_float(aug_info['crop_info'][1]) * ratio_x]
    edge_sides = [tf.to_float(aug_info['crop_info'][2]) * ratio_y,
                  tf.to_float(aug_info['crop_info'][3]) * ratio_x]
    H = tf.slice(H,
                 tf.concat([tf.to_int32(start_points), [0,]], axis=-1),
                 tf.concat([tf.to_int32(edge_sides), [num_channels,]], axis=-1))
    ## 2. Flip
    H = tf.cond(
      aug_info['whether_flip'],
      lambda: tf.image.flip_left_right(H),
      lambda: H)
  return H


def _get_other_items(provider, stuff, existing_items, new_items):
  res = []
  for item in new_items:
    if item in existing_items:
      res.append(stuff[existing_items.index(item)])
    else:
      res.append(provider.get([item])[0])
  return res


def get_input(provider, cfg, items):
  stuff = provider.get(items)
  if 'image' in items:
    img_pos = items.index('image')
    image = stuff[img_pos]
    # MPII has some huge images, which makes further processing too slow.
    # So, make image smaller if needed
    # IMP NOTE: Do not change the orig_im_ht or orig_im_wd, they are used for plotting
    # the pose and the pose is defined w.r.t to the original image size
    # Pose Label format: [16x3xn,] : x1,y1,score/isvisible...
    # if x1 and y1 are both -1, that point is not visible/labeled
    image = _resize_if_needed(image, cfg.MAX_INPUT_IMAGE_SIZE)
    if cfg.INPUT.INPUT_IMAGE_FORMAT.startswith('rendered-pose') or \
       cfg.INPUT.INPUT_IMAGE_FORMAT.startswith('pose-glimpse'):
      pose_label, orig_im_ht, orig_im_wd = _get_other_items(
        provider, stuff, items, ['pose', 'im_ht', 'im_wd'])
      # pose_label = tf.Print(pose_label, [pose_label], "Pose Label: ")
      pose_label_was_list = True
      if not isinstance(pose_label, list):
        pose_label_was_list = False
        pose_label = [pose_label]

      if cfg.INPUT.INPUT_IMAGE_FORMAT.startswith('rendered-pose'):
        rendered_pose = tf.stack([render_pose(
          pose_label[i], orig_im_ht, orig_im_wd,
          # TODO: the following tf.shape is going to read the image irrespective
          # of whether needed or not to compute shape. However the code isn't
          # slow so not worrying about it at the moment. But fix it.
          tf.cast(tf.shape(image)[-2], tf.int64),
          out_type=cfg.INPUT.INPUT_IMAGE_FORMAT_POSE_RENDER_TYPE) for
          i in range(len(pose_label))])
        rendered_pose = tf.image.resize_images(
          rendered_pose, tf.shape(image)[-3:-1])
        if not pose_label_was_list:
          rendered_pose = rendered_pose[0]
      else:
        image_glimpse = tf.stack([extract_glimpse(
          image, pose_label[i], orig_im_ht, orig_im_wd,
          cfg.TRAIN.IMAGE_SIZE if cfg.INPUT.POSE_GLIMPSE_RESIZE else -1,
          cfg.INPUT.POSE_GLIMPSE_CONTEXT_RATIO,
          cfg.INPUT.POSE_GLIMPSE_PARTS_KEEP) for
          i in range(len(pose_label))])


    if cfg.INPUT.INPUT_IMAGE_FORMAT.startswith('rendered-objects'):
      objects_label, orig_im_ht, orig_im_wd = _get_other_items(
        provider, stuff, items, ['objects', 'im_ht', 'im_wd'])
      # pose_label = tf.Print(pose_label, [pose_label], "Pose Label: ")
      rendered_objects = tf.stack([render_objects(
        objects_label[i], orig_im_ht, orig_im_wd,
        cfg.TRAIN.IMAGE_SIZE, out_channels=80) for
        i in range(len(objects_label))])

    # Final output
    if cfg.INPUT.INPUT_IMAGE_FORMAT == 'rendered-pose':
      image = rendered_pose
      # debugging
      # image = tf.tile(tf.reduce_mean(
      #   image, axis=-1, keep_dims=True), [1, 1, 1, 3])
    elif cfg.INPUT.INPUT_IMAGE_FORMAT == 'rendered-pose-on-image':
      image = tf.cast(tf.to_float(image) * 0.5 + \
                      tf.to_float(rendered_pose) * 0.5, tf.uint8)
    elif cfg.INPUT.INPUT_IMAGE_FORMAT == 'rendered-objects':
      image = rendered_objects
      # To debug
      # image = tf.cast(
      #   tf.to_float(image) * 0.0 + \
      #   tf.to_float(tf.image.resize_images(
      #     tf.reduce_mean(rendered_objects, axis=-1, keep_dims=True),
      #     tf.shape(image)[-3:-1])) * 1.0,
      #   tf.uint8)
    elif cfg.INPUT.INPUT_IMAGE_FORMAT == 'pose-glimpse':
      image = image_glimpse
    stuff[img_pos] = image
  return stuff


def train_preprocess_pipeline(provider, cfg, network_fn, num_pose_keypoints,
                              image_preprocessing_fn):

  [image, pose_label, orig_im_ht, orig_im_wd, action_label] = get_input(
    provider, cfg, ['image', 'pose', 'im_ht', 'im_wd', 'action_label'])
  # for consistency between video and image datasets, convert image datasets to
  # 1-frame videos
  if image.get_shape().ndims == 3:
    image = tf.expand_dims(image, 0)
    pose_label = [pose_label]
  train_image_size = cfg.TRAIN.IMAGE_SIZE or network_fn.default_image_size

  # joint preprocessing
  combined_preproc_flag = False
  with tf.name_scope('CombinedPreproc'):
    if num_pose_keypoints > 0 and not cfg.TRAIN.LOSS_FN_POSE == '':
      combined_preproc_flag = True
      all_pose_label_hmap = []
      all_pose_label_valid = []
      for pl in pose_label:
        pose_label_hmap, pose_label_valid = pose_to_heatmap(
          pl, orig_im_ht, orig_im_wd,
          # small enough for preproc, big enough to see
          max(200, cfg.TRAIN.FINAL_POSE_HMAP_SIDE),
          out_channels=num_pose_keypoints,
          # if needed, do using a conv layer with fixed kernel
          # would be faster on GPU
          do_gauss_blur=False,
          marker_wd_ratio=cfg.HEATMAP_MARKER_WD_RATIO)  # larger => large targets
        all_pose_label_hmap.append(pose_label_hmap)
        all_pose_label_valid.append(pose_label_valid)
      # concat on last axis for now (for preproc), will stack it (like the
      # valid labels) after that.
      pose_label_hmap = tf.concat(all_pose_label_hmap, axis=-1)
      pose_label_valid = tf.stack(all_pose_label_valid)

    # rgirdhar NOTE: This is the most expensive CPU part. My perf was super
    # slow with the output image sizes being 450x, because it'd first resize
    # the smallest dimension to 512 or so, and then take a 450 crop from that.
    # Doing that over RGB+heatmap channels was super slow, and is fixed when
    # using small sizes (now, 256 & 224 works well). Another issue was the
    # number of INTRA and INTER PARALLELIZATION THREADS, set in the train.py
    # which sped up a lot. Also saves from the machines getting stuck by
    # controlling the number of threads while giving better performance. For
    # me, the Inter=12 and Intra=4 worked well.
    preproc_info = {}
    # since images is 4D vector, need to reshape to pass it through preproc
    frames_per_video = image.get_shape().as_list()[0]
    image = tf.concat(tf.unstack(image), axis=-1)
    image = image_preprocessing_fn(
      image,
      train_image_size,
      train_image_size,
      resize_side_min=cfg.TRAIN.RESIZE_SIDE,
      resize_side_max=cfg.TRAIN.RESIZE_SIDE,
      preproc_info=preproc_info,
      modality=cfg.INPUT.VIDEO.MODALITY)  # works for image too, rgb by def
    image = tf.stack(tf.split(
      image, frames_per_video,
      axis=image.get_shape().ndims-1))
    if combined_preproc_flag:
      pose_label_hmap = _replay_augmentation(pose_label_hmap, preproc_info)
      pose_label_hmap = tf.image.convert_image_dtype(pose_label_hmap,
                                                     tf.float32)

      # undo any value scaling that happened while preproc
      pose_label_hmap -= tf.reduce_min(pose_label_hmap)
      pose_label_hmap /= (tf.reduce_max(pose_label_hmap) + cfg.EPS)
      # reduce the size of heatmaps to reduce memory usage in queues
      pose_label_hmap = tf.image.resize_images(
        pose_label_hmap,
        [cfg.TRAIN.FINAL_POSE_HMAP_SIDE,
         cfg.TRAIN.FINAL_POSE_HMAP_SIDE])
      pose_label_hmap.set_shape([
        pose_label_hmap.get_shape().as_list()[0],
        pose_label_hmap.get_shape().as_list()[1],
        num_pose_keypoints * frames_per_video])
      pose_label_hmap = tf.stack(tf.split(
        pose_label_hmap, frames_per_video,
        axis=pose_label_hmap.get_shape().ndims-1))
    else:
      pose_label_hmap = tf.zeros((0,))  # dummy value, not used
      pose_label_valid = tf.zeros((0,))  # dummy value, not used

  return image, pose_label_hmap, pose_label_valid, action_label
