import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from custom_ops.custom_ops_factory import render_objects

with tf.Session(''):
  T = render_objects(
    '1 1 0.743129 0.031770 0.151354 0.448363 0.994178\n'
    '1 1 0.813451 0.517574 0.303005 0.957526 0.975016',
    100,
    200,
    100,
    out_channels=80
  )
  A = T.eval()
  plt.imsave('temp.jpg', np.mean(A, axis=-1))
  import pdb
  pdb.set_trace()
  a = 1

