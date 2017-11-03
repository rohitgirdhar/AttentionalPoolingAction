# --------------------------------------------------------
# Written by Saurabh Gupta
# Modified by Ishan Misra
# rgirdhar: Obtained on March-09-2017 from
# https://github.com/imisra/latent-noise-icnm/blob/master/cap_eval_utils.py
# --------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d

from IPython.core.debugger import Tracer
import code

def calc_pr_ovr(counts, out, K):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
    K      : number of references
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """
  K = np.float64(K)
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts*(1.-1./K);
  fp = sortcounts.copy();
  for i in xrange(sortcounts.shape[0]):
    if sortcounts[i] > 1:
      fp[i] = 0.;
    elif sortcounts[i] == 0:
      fp[i] = 1.;
    elif sortcounts[i] == 1:
      fp[i] = 1./K;
  
  P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));

  # c = accumarray(sortcounts(:)+1, 1);
  c = [np.sum(np.array(sortcounts) == i) for i in xrange(int(max(sortcounts)+1))]
  ind = np.array(range(0, len(c)));
  numinst = ind*c*(K-1.)/K;
  numinst = np.sum(numinst, axis = 0)
  R = np.cumsum(tp)/numinst
  
  ap = voc_ap(R,P)
  return P, R, score, ap


def calc_pr_ovr_noref(counts, out):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
    K      : number of references
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """ 
  #binarize counts
  counts = np.array(counts > 0, dtype=np.float32);
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts;
  fp = sortcounts.copy();
  for i in xrange(sortcounts.shape[0]):
    if sortcounts[i] >= 1:
      fp[i] = 0.;
    elif sortcounts[i] < 1:
      fp[i] = 1.;
  P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));

  numinst = np.sum(counts);

  R = np.cumsum(tp)/numinst

  ap = voc_ap(R,P)
  return P, R, score, ap


def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Computes the AP under the precision recall curve.
  """

  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def compute_precision_score_mapping(thresh, prec, score):
  ind = np.argsort(thresh);
  thresh = thresh[ind];
  prec = prec[ind];
  for i in xrange(1, len(prec)):
    prec[i] = max(prec[i], prec[i-1]);
  
  indexes = np.unique(thresh, return_index=True)[1]
  indexes = np.sort(indexes);
  thresh = thresh[indexes]
  prec = prec[indexes]
  
  thresh = np.vstack((min(-1000, min(thresh)-1), thresh[:, np.newaxis], max(1000, max(thresh)+1)));
  prec = np.vstack((prec[0], prec[:, np.newaxis], prec[-1]));
  
  f = interp1d(thresh[:,0], prec[:,0])
  val = f(score)
  return val

def human_agreement(gt, K):
  """
  function [prec, recall] = human_agreement(gt, K)
  """
  c = np.zeros((K+1,1), dtype=np.float64)
  # namespace = globals().copy()
  # namespace.update(locals())
  # code.interact(local=namespace)

  for i in xrange(len(gt)):
    if gt[i]<K+1:
      c[gt[i]] += 1;
  #maxRun = len(gt);  
  # if len(gt) > K+1:
  #   print 'warning: '
  #   maxRun = K+1;
  # for i in xrange(maxRun):
  #   c[gt[i]] += 1;
  
  c = c/np.sum(c);
  ind = np.array(range(len(c)))[:, np.newaxis]

  n_tp = sum(ind*(ind-1)*c)/K;
  n_fp = c[1]/K;
  numinst = np.sum(c * (K-1) * ind) / K;
  prec = n_tp / (n_tp+n_fp);
  recall = n_tp / numinst;
  
  
  return prec, recall

#follows from http://arxiv.org/pdf/1312.4894v2.pdf (Sec 4.2)
def compute_warpstyle_pr(gtLabel, predMat, topK):
  assert gtLabel.shape == predMat.shape, 'gt {}; pred {}'.format(gtLabel.shape, predMat.shape)
  gtLabel = gtLabel.astype(np.float64)
  predMat = predMat.astype(np.float64)
  numTags = gtLabel.shape[1];
  numIm = gtLabel.shape[0];

  #first look at topK predictions per image
  topPreds = np.zeros_like(predMat);
  for imInd in range(numIm):
    topKInds = im_utils.maxk(predMat[imInd,...], topK);
    topPreds[imInd, topKInds] = 1;
  # tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
  gtLabel = (gtLabel > 0).astype(np.float64)
  topPreds = (topPreds > 0).astype(np.float64)
  corrMat = np.logical_and(gtLabel, topPreds).astype(np.float64)
  nc_per_tag = corrMat.sum(axis=0).astype(np.float64);
  ng_per_tag = gtLabel.sum(axis=0).astype(np.float64);
  np_per_tag = topPreds.sum(axis=0).astype(np.float64);
  #mean per-class
  perclass_recall = 0.0;
  perclass_precision = 0.0;
  eps = 1e-6;
  for t in range(numTags):
    cr = nc_per_tag[t]/(ng_per_tag[t]+eps);
    cp = nc_per_tag[t]/(np_per_tag[t]+eps);
    perclass_precision += cp;
    perclass_recall += cr;
  perclass_precision = (1.0/numTags) * perclass_precision;
  perclass_recall = (1.0/numTags) * perclass_recall;

  #overall
  overall_recall = nc_per_tag.sum()/(ng_per_tag.sum()+eps);
  overall_precision = nc_per_tag.sum()/(np_per_tag.sum()+eps);
  return perclass_precision, perclass_recall, overall_precision, overall_recall;

def print_benchmark_latex(evalFile, vocab = None, sortBy = "words", \
  printWords = False, printPos = True, printAgg = False, possOrder=None):
  #evalFile has the following ['details', 'agg', 'vocab', 'imdb'] 
  evalData = sg_utils.load_variables(evalFile);
  if vocab==None:
    vocab = evalData['vocab'];
  if 'details' in evalData:
    details = evalData['details'];
  else:
    details = evalData;
  ap = details['ap'];
  prec_at_human_rec = details['prec_at_human_rec'];
  human_prec = details['prec_at_human_rec'];
  words = vocab['words'];
  ind = 0;
  if possOrder is None:
    possOrder = ['NN', 'VB', 'JJ', 'DT', 'PRP', 'IN', 'other']
  print ' '.join(possOrder);
  for pos in possOrder:
    ind = [i for i,x in enumerate(vocab['poss']) if pos == x]
    ind = np.asarray(ind,dtype=np.int32)
    if any( np.isnan(ap[0,ind] )):
       #print 'nan numbers ... skipping them for mean'
       print 'nan numbers ... setting them to zero for mean stats'
       ap[0, ind[np.where(np.isnan(ap[0, ind]))]] = 0;
    print '%.1f &'%(100*np.mean(ap[0,ind])),
  print '%.1f & &'%(100*np.mean(ap[0, :]))
  for pos in possOrder:
    ind = [i for i,x in enumerate(vocab['poss']) if pos == x]
    ind = np.asarray(ind,dtype=np.int32)
    if any( np.isnan(prec_at_human_rec[0,ind] )) or \
       any( np.isnan(human_prec[0,ind] )) :
       #print 'nan numbers ... skipping them for mean'
       print 'nan numbers ... setting them to zero for mean stats'
       prec_at_human_rec[0, ind[np.where(np.isnan(prec_at_human_rec[0, ind]))]] = 0;
       human_prec[0, ind[np.where(np.isnan(human_prec[0, ind]))]] = 0;
    print '%.1f &'%(100*np.mean(prec_at_human_rec[0,ind])),
  print '%.1f \\\\'%(100*np.mean(prec_at_human_rec[0, :]))
  



def print_benchmark_plain(evalFile, vocab = None, \
  sortBy = "words", printWords = False, printPos = True, printAgg = False):
  #evalFile has the following ['details', 'agg', 'vocab', 'imdb'] 
  evalData = sg_utils.load_variables(evalFile);
  if vocab==None:
    vocab = evalData['vocab'];
  if 'details' in evalData:
    details = evalData['details'];
  else:
    details = evalData;
  ap = details['ap'];
  prec_at_human_rec = details['prec_at_human_rec'];
  human_prec = details['prec_at_human_rec'];
  words = vocab['words'];
  ind = 0;

  if sortBy == "words":
    srtInds = np.argsort(words);
  elif sortBy == "ap":
    srtInds = np.argsort(ap);
    srtInds = srtInds[0];
    srtInds = srtInds[::-1];
  if printWords == True:
    print "{:>50s}".format("-"*50)
    print "{:^50s}".format("Word metrics")
    print "{:>50s}".format("-"*50)
    print "{:>15s} {:>8s} {:>6s} :     {:^5s}     {:^5s}". \
      format("Words","POS","Counts","mAP", "p@H")
    for i in srtInds:
      print "{:>15s} {:>8s} {:6d} :     {:5.2f}     {:5.2f}". \
        format(words[i], vocab['poss'][i], vocab['counts'][i], 100*np.mean(ap[0, i]), 100*np.mean(prec_at_human_rec[0, i]));

  if printPos:
    print "{:>50s}".format("-"*50)
    print "{:^50s}".format("POS metrics")
    print "{:>50s}".format("-"*50)
    print "{:>15s} :     {:^5s}     {:^5s}     {:^5s}". \
    format("POS", "mAP", "p@H", "h")

    for pos in list(set(vocab['poss'])):
      ind = [i for i,x in enumerate(vocab['poss']) if pos == x]
      ind = np.asarray(ind)
      if any( np.isnan(ap[0,ind] )) or \
         any( np.isnan(prec_at_human_rec[0,ind] )) or \
         any( np.isnan(human_prec[0,ind] )) :
         print 'nan numbers ... setting them to zero for mean stats'
         ap[0, ind[np.where(np.isnan(ap[0, ind]))]] = 0;
         prec_at_human_rec[0, ind[np.where(np.isnan(prec_at_human_rec[0, ind]))]] = 0;
         human_prec[0, ind[np.where(np.isnan(human_prec[0, ind]))]] = 0;
      print "{:>11s} [{:4d}]:     {:5.2f}     {:5.2f}     {:5.2f}". \
        format(pos, len(ind), 100*np.mean(ap[0, ind]), 100*np.mean(prec_at_human_rec[0, ind]), \
        100*np.mean(human_prec[0, ind]))

  if printAgg:
    print "{:>50s}".format("-"*50)
    print "{:^50s}".format("Agg metrics")
    print "{:>50s}".format("-"*50)
    print "{:>15s} :     {:^5s}     {:^5s}     {:^5s}". \
      format("agg", "mAP", "p@H", "h")
    pos = 'all';
    ind = srtInds;
    ind = np.asarray(ind);
    if any( np.isnan(ap[0,ind] )) or \
         any( np.isnan(prec_at_human_rec[0,ind] )) or \
         any( np.isnan(human_prec[0,ind] )) :
         print 'nan numbers ... setting them to zero for mean stats'
         ap[0, ind[np.where(np.isnan(ap[0, ind]))]] = 0;
         prec_at_human_rec[0, ind[np.where(np.isnan(prec_at_human_rec[0, ind]))]] = 0;
         human_prec[0, ind[np.where(np.isnan(human_prec[0, ind]))]] = 0;
    print "{:>11s} [{:^4d}]     :     {:^5.2f}     {:5.2f}     {:5.2f}". \
      format(pos, len(ind), 100*np.mean(ap[0, ind]), 100*np.mean(prec_at_human_rec[0, ind]), \
        100*np.mean(human_prec[0, ind]))
