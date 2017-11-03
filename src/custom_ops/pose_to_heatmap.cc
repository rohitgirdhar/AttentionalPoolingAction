#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <tuple>

#include <opencv2/opencv.hpp>

using namespace tensorflow;
using namespace std;

REGISTER_OP("PoseToHeatmap")
  .Attr("out_channels: int = 16")
  .Attr("marker_wd_ratio: float = 0.1")
  .Attr("do_gauss_blur: bool = True")
  .Input("pose_label: int64")
  .Input("im_ht: int64")
  .Input("im_wd: int64")
  .Input("out_wd: int64")  // out_height decided using this and aspect ratio of image
  .Output("heatmap: float")
  .Output("is_valid: bool");  // a bit for each channel, if that pose label is valid or not

class PoseToHeatmapOp : public OpKernel {
 public:
  explicit PoseToHeatmapOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("out_channels", &out_channels_));
    OP_REQUIRES_OK(
        context, context->GetAttr("marker_wd_ratio", &marker_wd_ratio_));
    OP_REQUIRES_OK(
        context, context->GetAttr("do_gauss_blur", &do_gauss_blur_));
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

    // The pose label should be 16 keypoints, with X,Y,is_visible
    int num_keypoints = out_channels_;
    assert(pose_label.size() % (3 * num_keypoints) == 0);
    int n_rects = pose_label.size() / (3 * num_keypoints);

    // Create output tensors
    TensorShape out_shape {out_ht, out_wd, out_channels_};
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        context, 
        context->allocate_output(
          0,
          out_shape,
          &output_tensor));
    auto output = output_tensor->tensor<float, 3>();
    TensorShape out_shape_valid {out_channels_};
    Tensor* output_tensor_valid = NULL;
    OP_REQUIRES_OK(
        context, 
        context->allocate_output(
          1,
          out_shape_valid,
          &output_tensor_valid));
    auto output_valid = output_tensor_valid->tensor<bool, 1>();

    int elts_per_pose = num_keypoints * 3;
    for (int i = 0; i < num_keypoints; i++) {
      cv::Mat channel(out_ht, out_wd, CV_32FC1, 0.0);
      output_valid(i) = false;
      for (int rid = 0; rid < n_rects; rid++) {  // for each rectangle
        int x = pose_label(rid * elts_per_pose + i * 3) * out_wd / im_wd;
        int y = pose_label(rid * elts_per_pose + i * 3 + 1) * out_ht / im_ht;
        int is_visible = pose_label(rid * elts_per_pose + i * 3 + 2);  // ignore this
        if (pose_label(rid * elts_per_pose + i * 3) >= 0 &&
            pose_label(rid * elts_per_pose + i * 3 + 1) >= 0) {
          output_valid(i) = true;
          circle(channel, cv::Point(x, y),
                 (int) out_wd * marker_wd_ratio_,
                 cv::Scalar(1.0, 1.0, 1.0), -1);
          if (do_gauss_blur_)
            GaussianBlur(channel, channel, cv::Size(7, 7), 0);
        }
      }
      for (int r = 0; r < channel.rows; r++) {
        for (int c = 0; c < channel.cols; c++) {
          output(r, c, i) = channel.at<float>(r, c);
        }
      }
    }
  }
  
 private:
  int out_channels_;
  float marker_wd_ratio_;
  bool do_gauss_blur_;
};

REGISTER_KERNEL_BUILDER(Name("PoseToHeatmap").Device(DEVICE_CPU), PoseToHeatmapOp);
