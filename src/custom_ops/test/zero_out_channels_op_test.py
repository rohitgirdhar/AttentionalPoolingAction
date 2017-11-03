import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from custom_ops.custom_ops_factory import zero_out_channels

with tf.Session(''):
  A = np.ones((1, 3, 3, 5))
  channels = [True, False, True, True, True]
  B = zero_out_channels(A, channels)
  print B
  C = B.eval()
  assert(np.all(C[:, :, :, 0] == 1))
  assert(np.all(C[:, :, :, 1] == 0))
  assert(np.all(C[:, :, :, 2] == 1))
  assert(np.all(C[:, :, :, 3] == 1))
  import pdb
  pdb.set_trace()
  a = 1

