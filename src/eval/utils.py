from eval.cap_eval_utils import calc_pr_ovr_noref
import numpy as np

def compute_map(all_logits, all_labels):
  num_classes = all_logits.shape[1]
  APs = []
  for cid in range(num_classes):
    this_logits = all_logits[:, cid]
    this_labels = (all_labels == cid).astype('float32')
    if np.sum(this_labels) == 0:
      print('No positive videos for class {}. Ignoring...'.format(cid))
      continue
    _, _, _, ap = calc_pr_ovr_noref(this_labels, this_logits)
    APs.append(ap)
  mAP = np.mean(APs)
  return mAP, APs
