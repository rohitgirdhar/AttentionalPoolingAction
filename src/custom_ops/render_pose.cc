#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <tuple>

#include <opencv2/opencv.hpp>

#include "pose_utils.hpp"

using namespace tensorflow;
using namespace std;
namespace pt = boost::property_tree;


REGISTER_OP("RenderPose")
  .Attr("marker_wd_ratio: float = 0.01")  // ratio of output image width
  .Attr("out_type: int = 1")  // RENDER_POSE_OUT_TYPE_RGB or RENDER_POSE_OUT_TYPE_SPLITCHANNEL
  .Input("pose_label: int64")
  .Input("im_ht: int64")
  .Input("im_wd: int64")
  .Input("out_wd: int64")
  .Output("image: float");

class RenderPoseOp : public OpKernel {
 public:
  explicit RenderPoseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("marker_wd_ratio", &marker_wd_ratio_));
    OP_REQUIRES_OK(
        context, context->GetAttr("out_type", &out_type_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& pose_label_tensor = context->input(0);
    auto pose_label = pose_label_tensor.flat<long long>();
    const Tensor& im_ht_tensor = context->input(1);
    auto im_ht = im_ht_tensor.flat<long long>()(0);
    const Tensor& im_wd_tensor = context->input(2);
    auto im_wd = im_wd_tensor.flat<long long>()(0);
    const Tensor& out_wd_tensor = context->input(3);
    auto out_wd = out_wd_tensor.flat<long long>()(0);
    int out_ht = ((im_ht * out_wd * 1.0) / im_wd);

    int num_keypoints = 16;  // MPII poses
    assert(pose_label.size() % (3 * num_keypoints) == 0);
    int n_people = pose_label.size() / (3 * num_keypoints);
    vector<vector<tuple<float,float,float>>> poses;
    int elts_per_pose = 3 * num_keypoints;
    for (int i = 0; i < n_people; i++) {
      vector<tuple<float,float,float>> person;
      for (int j = 0; j < num_keypoints; j++) {
        int x = pose_label(elts_per_pose * i + 3 * j);
        int y = pose_label(elts_per_pose * i + 3 * j + 1);
        int is_visible = pose_label(elts_per_pose * i + 3 * j + 2);
        // TODO (rgirdhar): Maybe this needs be fixed
        if (x == -1 && y == -1) {
          is_visible = 0;
        } else {
          is_visible = 1;
        }
        person.push_back(make_tuple(x, y, is_visible));
      }
      poses.push_back(convert_pose_mpii_to_coco(person));
    }

    cv::Mat render = render_pose(
        poses, out_ht, out_wd,
        im_ht, im_wd, out_wd * marker_wd_ratio_,
        out_type_);
    // Create an output tensor
    TensorShape out_shape {out_ht, out_wd, render.channels()};
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
          0,
          out_shape,
          &output_tensor));
    auto output = output_tensor->tensor<float, 3>();

    for (int i = 0; i < render.rows; i++) {
      for (int j = 0; j < render.cols; j++) {
        float *pixel = render.ptr<float>(i, j);
        for (int k = 0; k < render.channels(); k++) {
          output(i, j, k) = pixel[render.channels()-k-1];
        }
      }
    }
  }

 private:

  vector<tuple<float,float,float>> convert_pose_mpii_to_coco(
      vector<tuple<float,float,float>> poses) {
    // Using the coco definition from https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose
    // Using the MPII definition from http://human-pose.mpi-inf.mpg.de/#download
    vector<tuple<float,float,float>> res;
    auto dummy = make_tuple(0, 0, 0);  // for the parts I don't have in MPII
    map<int, int> coco_to_mpii = {
      {0, 9},  // Nose, head_top (approx)
      {1, 8},
      {2, 12},
      {3, 11},
      {4, 10},
      {5, 13},
      {6, 14},
      {7, 15},
      {8, 2},
      {9, 1},
      {10, 0},
      {11, 3},
      {12, 4},
      {13, 5},
      {14, -1},
      {15, -1},
      {16, -1},
      {17, -1},
      {18, -1}
    };
    for (int i = 0; i < coco_to_mpii.size(); i++) {
      if (coco_to_mpii[i] == -1) {
        res.push_back(dummy);
      } else {
        res.push_back(poses[coco_to_mpii[i]]);
      }
    }
    return res;
  }

  float marker_wd_ratio_;
  int out_type_;
};

REGISTER_KERNEL_BUILDER(Name("RenderPose").Device(DEVICE_CPU), RenderPoseOp);
