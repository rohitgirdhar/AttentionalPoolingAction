import tensorflow as tf

# TODO: move this to the main train script if useful. Not a good idea to have this inside.
tf.app.flags.DEFINE_string(
    'pose_style', 'heatmap',
    'Select style for pose to be rendered [heatmap/render].')
FLAGS = tf.app.flags.FLAGS

IM_HT = 256
IM_WD = 340

def _get_frame_sublist(start_frame, duration, num_samples, num_consec_frames,
                       randomFromSegmentStyle=None):
  # follow segmental architecture
  res = []
  step = tf.cast((duration - tf.constant(num_consec_frames)) / 
                 (tf.constant(num_samples)), 'int32')
  step = tf.maximum(step, 1)
  cur_end_point = 0
  if randomFromSegmentStyle is None:
    if num_samples == 1:
      randomFromSegmentStyle = True  # because otherwise would not make sense
    else:
      randomFromSegmentStyle = False
  # start_frame = tf.Print(start_frame, [start_frame], 'Using start frame: ')
  # The following will be printed as many times as the number of read threads
  if randomFromSegmentStyle:
    tf.logging.info('Reading in random segment style')
  else:
    tf.logging.info('IMP NOTE:: Reading uniform frames')
  for i in range(num_samples):
    if randomFromSegmentStyle:
      res.append(tf.random_uniform([1],
                                   tf.minimum(start_frame + step * i,
                                              duration-num_consec_frames-1),
                                   tf.minimum(start_frame + step * (i+1),
                                              duration-num_consec_frames),
                                   dtype='int32')[0])
    else:
      res.append(tf.minimum(start_frame + step * i, duration - 1))
  # To debug
  # res[0] = tf.Print(res[0], res, 'Offsets:' )
  [el.set_shape(()) for el in res]
  return res

def _get_frame_sublist_SAME_AS_CAFFE(
  start_frame, duration, num_samples, num_consec_frames,
  randomFromSegmentStyle=None):
  # follow segmental architecture
  res = []
  avg_duration = tf.cast(duration / tf.constant(num_samples), 'int32')
  cur_end_point = 0
  if randomFromSegmentStyle is None:
    if num_samples == 1:
      randomFromSegmentStyle = True  # because otherwise would not make sense
    else:
      randomFromSegmentStyle = False
  # start_frame = tf.Print(start_frame, [start_frame], 'Using start frame: ')
  # The following will be printed as many times as the number of read threads
  if randomFromSegmentStyle:
    tf.logging.info('Reading in random segment style')
  else:
    tf.logging.info('IMP NOTE:: Reading uniform frames')
  for i in range(num_samples):
    if randomFromSegmentStyle:
      offset = tf.random_uniform([1], 0, avg_duration-num_consec_frames+1,
                                 dtype=tf.int32)
      T = tf.cond(tf.greater_equal(avg_duration, num_consec_frames),
                  lambda: offset + i * avg_duration,
                  lambda: tf.constant([1]))
      res.append(T[0])
    else:
      T = tf.cond(tf.greater_equal(avg_duration, num_consec_frames),
                  lambda: (
                    avg_duration-num_consec_frames+1)/2 + i*avg_duration,
                  lambda: tf.constant([1]))
      res.append(T[0])
  # To debug
  # res[0] = tf.Print(res[0], res, 'Offsets:' )
  return res

def _read_from_disk_spatial(fpath, nframes, num_samples=25, start_frame=0,
                            file_prefix='', file_zero_padding=4, file_index=1,
                            dataset_dir='', frame_sublist=None,
                            randomFromSegmentStyle=None):
    if frame_sublist is None:
      frame_sublist = _get_frame_sublist(start_frame, nframes, num_samples, 1,
                                        randomFromSegmentStyle)
    allimgs = []
    with tf.variable_scope('read_rgb_video'):
        for i in range(num_samples):
            with tf.variable_scope('read_rgb_image'):
                prefix = file_prefix + '_' if file_prefix else ''
                impath = tf.string_join([
                    tf.constant(dataset_dir + '/'),
                    fpath, tf.constant('/'),
                    prefix,
                    tf.as_string(frame_sublist[i] + file_index,
                      width=file_zero_padding, fill='0'),
                    tf.constant('.jpg')])
                # To debug
                # impath = tf.Print(impath, [impath], message='Reading image:')
                img_str = tf.read_file(impath)
            allimgs.append(img_str)
    return allimgs


def _read_from_disk_temporal(
    fpath, nframes, num_samples=25,
    optical_flow_frames=10, start_frame=0,
    file_prefix='', file_zero_padding=4, file_index=1,
    dataset_dir='', frame_sublist=None, randomFromSegmentStyle=None):
    if frame_sublist is None:
      frame_sublist = _get_frame_sublist(start_frame, nframes, num_samples,
                                         optical_flow_frames,
                                         randomFromSegmentStyle)
    allimgs = []
    with tf.variable_scope('read_flow_video'):
        for i in range(num_samples):
            with tf.variable_scope('read_flow_image'):
              flow_img = []
              for j in range(optical_flow_frames):
                # To protect for small videos, avoid overshooting the filelist
                frame_id = frame_sublist[i] + j
                frame_id = tf.cond(
                  tf.greater(frame_id, nframes-2),
                  lambda: nframes-2,
                  lambda: frame_id)

                with tf.variable_scope('read_flow_channels'):
                  for dr in ['x', 'y']:
                    prefix = file_prefix + '_' if file_prefix else ''
                    impath = tf.string_join([
                        tf.constant(dataset_dir + '/'),
                        fpath, tf.constant('/'),
                        prefix, '%s_' % dr,
                        tf.as_string(frame_id + file_index,
                          width=file_zero_padding, fill='0'),
                        tf.constant('.jpg')])
                    # impath = tf.Print(impath, [impath], "Read file: ")
                    img_str = tf.read_file(impath)
                    flow_img.append(img_str)
              allimgs.append(flow_img)
    return allimgs


def _read_from_disk_pose(
    fpath, nframes, num_samples=25,
    pose_frames=5, start_frame=0,
    file_prefix='', file_zero_padding=4, file_index=1,
    dataset_dir='', frame_sublist=None, randomFromSegmentStyle=None,
    file_ext='.jpg'):
    from custom_ops.custom_ops_factory import read_file_safe
    if frame_sublist is None:
      frame_sublist = _get_frame_sublist(start_frame, nframes, num_samples,
                                         pose_frames,
                                         randomFromSegmentStyle)
    allimgs = []
    with tf.variable_scope('read_pose_video'):
      for i in range(num_samples):
        with tf.variable_scope('read_pose_image'):
          pose_img = []
          for j in range(pose_frames):
            # To protect for small videos, avoid overshooting the filelist
            frame_id = frame_sublist[i] + j
            frame_id = tf.cond(
              tf.greater(frame_id, nframes-1),  # there are nframes-1 flow
              lambda: nframes-1,
              lambda: frame_id)

            prefix = file_prefix + '_' if file_prefix else ''
            impath = tf.string_join([
              tf.constant(dataset_dir + '/'),
              fpath, tf.constant('/'),
              prefix,
              tf.as_string(frame_id + file_index,
              width=file_zero_padding, fill='0'),
              tf.constant(file_ext)])
            # img_str = tf.read_file(impath)
            img_str = read_file_safe(impath)
            pose_img.append(img_str)
          allimgs.append(pose_img)
    return allimgs


def decode_rgb(img_str):
  with tf.variable_scope('decode_rgb_frame'):
    img = tf.image.decode_jpeg(img_str, channels=3)
    # Always convert before resize, this is a bug in TF
    # https://github.com/tensorflow/tensorflow/issues/1763
    # IMPORTANT NOTE: The original netvlad model was trained with the convert
    # happening after the resize, and hence it's trained with the large values.
    # It still works if I do that, but I'm training a new netvlad RGB model
    # with the current setup.
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  return [img]


def decode_flow(img_str, perImageChannels=1):
  # IMPORTANT NOTE: I am now resizing the flow frames before running through
  # the preprocessing. I was not doing that earlier (in the master). This leads
  # to the 66 number to drop to 63 on HMDB. But it should be fixable by
  # re-training with this setup
  with tf.variable_scope('decode_flow_frame'):
    img = tf.concat([tf.image.decode_jpeg(el, channels=perImageChannels)
      for el in tf.unstack(img_str)], axis=2)
    # Always convert before resize, this is a bug in TF
    # https://github.com/tensorflow/tensorflow/issues/1763
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  return [img]


def decode_poseJson(img_str, perImageChannels=1):
  from custom_ops.custom_ops_factory import json_to_pose
  with tf.variable_scope('decode_poseJson_frame'):
    pose_style = FLAGS.pose_style
    img = tf.concat([json_to_pose(
      el, out_height=IM_HT, out_width=IM_WD,
      marker_wid=5 if pose_style=='render' else 20,
      out_style=pose_style)
      for el in img_str], axis=2)
    # img = tf.image.resize_images(img, [IM_HT, IM_WD]) # not any faster
    # TODO: remove the following checks once sure
    # with tf.control_dependencies(
    #   [tf.assert_less_equal(img, tf.constant(1.5)),
    #    tf.assert_greater_equal(img, tf.constant(-0.5))]):
    #   img = tf.identity(img)
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  return [img]


def _decode_from_string(img_str, modality):
  if modality == 'rgb':
    img = decode_rgb(img_str)
  elif modality.startswith('flow'):
    img = decode_flow(img_str)
  elif modality.startswith('rgb+flow'):
    with tf.name_scope('decode_rgbNflow'):
      img_rgb = decode_rgb(img_str[..., 0])
      img_flow = decode_flow(img_str[..., 1:])
      img = [img_rgb[0], img_flow[0]]
  elif modality.startswith('posejson'):
    img = decode_poseJson(img_str)
  elif modality.startswith('pose'):
    img = decode_flow(img_str, perImageChannels=3)
  im_ht = tf.reduce_max([tf.shape(el)[-3] for el in img])
  im_wd = tf.reduce_max([tf.shape(el)[-2] for el in img])
  img = [tf.image.resize_images(el, [IM_HT, IM_WD]) for el in img]
  return img, im_ht, im_wd
