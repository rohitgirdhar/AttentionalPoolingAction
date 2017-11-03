import numpy as np
import h5py

from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import var_name_mapper


def restore_model(checkpoint_path,
                  variables_to_restore,
                  ignore_missing_vars=False,
                  var_name_mapper_type=None):
  all_ops = []
  checkpoint_variables = variables_to_restore
  if checkpoint_path.endswith('.npy'):
    vars_to_restore_names = [
      el.name for el in checkpoint_variables]
    key_name_mapper = var_name_mapper.map(var_name_mapper_type)
    init_weights = np.load(checkpoint_path).item()
    init_weights_final = {}
    vars_restored = []
    for key in init_weights.keys():
      for subkey in init_weights[key].keys():
        final_key_name = key_name_mapper(
          key + '/' + subkey)
        if final_key_name not in vars_to_restore_names:
          logging.info('Not using %s from npy' % final_key_name)
          continue
        target_shape = slim.get_model_variables(
          final_key_name)[0].get_shape().as_list()
        pretrained_wts = init_weights[key][subkey].copy()
        target_shape_squeezed = np.delete(
          target_shape, np.where(np.array(target_shape) == 1))
        pretrained_shape_squeezed = np.delete(
          pretrained_wts.shape, np.where(np.array(pretrained_wts.shape) == 1))

        go_ahead = False  # whether or not I'll be able to copy these weights
        if np.any(target_shape_squeezed !=
                  pretrained_shape_squeezed):
          logging.info('Shape mismatch var: %s from npy [%s vs %s]. ' % (
                       final_key_name, target_shape,
                       pretrained_wts.shape))
          if pretrained_shape_squeezed[-2] != target_shape_squeezed[-2]:
            logging.info('Trying repeating channels to make it similar.')
            pretrained_wts = np.repeat(
              np.mean(pretrained_wts, axis=-2, keepdims=True),
              repeats=target_shape_squeezed[-2],
              axis=-2)
            if np.all(target_shape_squeezed == pretrained_wts.shape):
              logging.info('Success! Copying the hacked weights.')
              go_ahead = True
            else:
              logging.info('Couldnot match the weights still.')
        else:
          go_ahead = True
        if go_ahead:
          init_weights_final[final_key_name] = \
            pretrained_wts
          vars_restored.append(final_key_name)
    init_weights = init_weights_final
    for v in vars_to_restore_names:
      if v not in vars_restored:
        logging.fatal('No weights found for %s' % v)
        if not ignore_missing_vars:
          raise ValueError()
    all_ops.append(slim.assign_from_values_fn(init_weights))
  else:
    all_ops.append(assign_from_checkpoint_fn(
      checkpoint_path, checkpoint_variables,
      ignore_missing_vars=ignore_missing_vars,
      resize_variables=True))
  def combined(sess):
    for op in all_ops:
      op(sess)
  return combined


def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False, resize_variables=False):
  """Modified function from
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/framework/python/ops/variables.py
  Mod by rgirdhar to allow for repeating the channels dimension in case a layer
  does not match. It's useful for setting the first layer in flow models for
  videos. Does this only when resize_variables is True.
  """
  """Returns a function that assigns specific variables from a checkpoint.

  If ignore_missing_vars is True and no variables are found in the checkpoint
  it returns None.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
        use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
        checkpoint to the corresponding variables to initialize. If empty or
        `None`, it would return `no_op(), None`.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
        the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
        which are of different shape then the ones stored in the checkpoint but
        which have the same number of elements.
    resize_variables: Boolean, if True it would repeat the channels to match
        the target variable dimensions

  Returns:
    A function that takes a single argument, a `tf.Session`, that applies the
    assignment operation. If no matching variables were found in the checkpoint
    then `None` is returned.

  Raises:
    ValueError: If var_list is empty.
  """
  if not var_list:
    raise ValueError('var_list cannot be empty')
  reader = pywrap_tensorflow.NewCheckpointReader(model_path)
  if isinstance(var_list, dict):
    var_dict = var_list
  else:
    var_dict = {var.op.name: var for var in var_list}
  available_vars = {}
  for var in var_dict:
    if reader.has_tensor(var):
      go_ahead = False
      V = reader.get_tensor(var)
      ckpt_shape = list(V.shape)
      target_shape = var_dict[var].get_shape().as_list()
      if np.all(ckpt_shape == target_shape):
        go_ahead = True
      else:
        if resize_variables:
          logging.warning('Resizing to assign to variable {} to {} from {}'.format(
            var, var_dict[var].get_shape().as_list(),
            V.shape))
          V = np.repeat(
            np.mean(V, axis=-2, keepdims=True),
            repeats=target_shape[-2],
            axis=-2)
          ckpt_shape = list(V.shape)
          if np.all(ckpt_shape == target_shape):
            logging.info('Was able to match shape, so restoring the var :-)')
            go_ahead = True
          else:
            logging.error('Was not able to match shape, not restoring it!!!')
            go_ahead = False
        else:
          logging.error('Found a shape mismatch. Set resize_var to true to '
                        'do a hacky shape copy.')
      if go_ahead:
        available_vars[var] = V
    else:
      logging.warning(
          'Variable %s missing in checkpoint %s', var, model_path)
      if not ignore_missing_vars:
        raise ValueError()
  return slim.assign_from_values_fn(available_vars)


def get_special_assigns(special_assign_vars):
  init_wts = {}
  special_assign_vars = special_assign_vars.split(',')
  for i in range(len(special_assign_vars) / 2):
    var_name = special_assign_vars[2*i]
    file_path = special_assign_vars[2*i+1]
    with h5py.File(file_path, 'r') as fin:
      init_wts[var_name] = fin['feat'].value
    logging.info('Special Assign: %s with a %s array' % (
      var_name, init_wts[var_name].shape))
  return slim.assign_from_values_fn(init_wts)
