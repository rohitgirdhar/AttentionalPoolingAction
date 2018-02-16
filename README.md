# Attentional Pooling for Action Recognition

If this code helps with your work/research, please consider citing

Rohit Girdhar and Deva Ramanan. **Attentional Pooling for Action Recognition**. Advances in Neural Information Processing Systems (NIPS), 2017.

```txt
@inproceedings{Girdhar_17b_AttentionalPoolingAction,
    title = {Attentional Pooling for Action Recognition},
    author = {Girdhar, Rohit and Ramanan, Deva},
    booktitle = {NIPS},
    year = 2017
}
```

## Pre-requisites

This code was trained and tested with

1. CentOS 6.5
2. Python 2.7
3. TensorFlow 1.1.0-rc2 ([6a1825e2](https://github.com/tensorflow/tensorflow/tree/6a1825e2369d2537e15dc585705c53c4b763f3f6))

## Getting started

Clone the code and create some directories for outputs

```bash
$ git clone --recursive
$ export ROOT=`pwd`/AttentionalPoolingAction
$ cd $ROOT/src/
$ mkdir -p expt_outputs data
$ # compile some custom ops
$ cd custom_ops; make; cd ..
```

## Data setup

You can download the `tfrecord` files for MPII I used from
[here](https://cmu.box.com/shared/static/xb7esevyl6uzmra2eehnkbt2ud7awld9.tar)
and uncompress on to a fast local disk.
If you want to create your own tfrecords, you can use the following steps, which is
what I used to create the linked tfrecord files

Convert the MPII data into tfrecords. The system also can read from individual JPEG files,
but that needs a slightly different intial setup.

First download the MPII [images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)
and [annotations](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip),
and un-compress the files.

```bash
$ cd $ROOT/utils/dataset_utils
$ # Set the paths for MPII images and annotations file in gen_tfrecord_mpii.py
$ python gen_tfrecord_mpii.py  # Will generate the tfrecord files
```

### Keypoint labels for other datasets

While MPII dataset comes with pose labels, I also experiment with HMDB-51 and HICO, pose for which was computed using an initial vesion of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). I provide the extracted keypoints here: [HMDB51]() and [HICO](https://cmu.box.com/shared/static/42xizpt0w3almdgwczjxawvc1pvpesoa.tar).

## Testing pre-trained models

First download and unzip the
[pretrained models](https://cmu.box.com/shared/static/s72scgtjj3lm60hsufi25rfjs2dk3a7i.zip)
to a `$ROOT/src/pretrained_models/`.
The models can be run by

```bash
# Baseline model (no attention)
$ python eval.py --cfg ../experiments/001_MPII_ResNet_pretrained.yaml
# With attention
$ python eval.py --cfg ../experiments/002_MPII_ResNet_pretrained.yaml
# With pose regularized attention
$ python eval.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention_pretrained.yaml
```

### Expected performance on MPII Validation set

| Method  | mAP | Accuracy |
|--------|-----|------|
| Baseline (no attention) | 26.2 | 33.5 |
| With attention | 30.3 | 37.2 |
| With pose regularized attention | 30.6 | 37.8 |

## Training

Train a attentional pooled model on MPII dataset, using `python train.py --cfg <path to YAML file>`.

```bash
$ cd $ROOT/src
$ python train.py --cfg ../experiments/002_MPII_ResNet_withAttention.yaml
# To train the model with pose regularized attention, use the following config
$ python train.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention.yaml
# To train the baseline without attention, use the following config
$ python train.py --cfg ../experiments/001_MPII_ResNet.yaml
```

## Testing and evaluation

Test the model trained above on the validation set, using `python eval.py --cfg <path to YAML file>`.

```bash
$ python eval.py --cfg ../experiments/002_MPII_ResNet_withAttention.yaml
# To evaluate the model with pose regularized attention
$ python eval.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention.yaml
# To evaluate the model without attention
$ python train.py --cfg ../experiments/001_MPII_ResNet.yaml
```

The performance of these models should be similar to the above
released pre-trained models.

## Train + test on the final test set

This is for getting the final number on MPII test set.

```bash
# Train on the train + val set
$ python train.py --cfg ../experiments/002_MPII_ResNet_withAttention_train+val.yaml
# Test on the test set
$ python eval.py --cfg ../experiments/002_MPII_ResNet_withAttention_train+val.yaml --save
# Convert the output into the MAT files as expected by MPII authors (requires matlab/octave)
$ cd ../utils;
$ bash convert_mpii_result_for_eval.sh ../src/expt_outputs/002_MPII_ResNet_withAttention_train+val.yaml/<filename.h5>
# Now the generated mat file can be emailed to MPII authors for test evaluation
```
