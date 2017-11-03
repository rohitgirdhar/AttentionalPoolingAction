import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from custom_ops.custom_ops_factory import pose_to_heatmap

with tf.Session(''):
  pose = [50, 50, 1] * 3 +\
      [0, 0, 1] * 2 +\
      [-1, -1, 1] * 11
  pose += [90, 90, 1] * 3 +\
      [0, 0, 1] * 2 +\
      [-1, -1, 1] * 11

  T, T_valid = pose_to_heatmap(
    pose,
    100,
    200,
    100,
    out_channels=16
  )
  A = T.eval()
  A_valid = T_valid.eval()
  plt.imsave('temp.jpg', np.mean(A, axis=-1))
  print A_valid
  import pdb
  pdb.set_trace()
  a = 1

