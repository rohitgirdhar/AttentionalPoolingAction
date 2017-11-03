#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <tuple>

#include <opencv2/opencv.hpp>

using namespace tensorflow;
using namespace std;

REGISTER_OP("RenderObjects")
  .Attr("out_channels: int = 80")
  .Input("objects_label: string")
  .Input("im_ht: int64")
  .Input("im_wd: int64")
  .Input("out_wd: int64")  // out_height decided using this and aspect ratio of image
  .Output("image: float");


void read_detections(
    string objects_label,
    vector<tuple<int,float,float,float,float,float>> &detections) {
  istringstream ss(objects_label);
  int ob_label, id;  // ignore the id
  float conf, xmin, ymin, xmax, ymax;
  detections.clear();
  while (ss >> id >> ob_label >> conf >> xmin >> ymin >> xmax >> ymax) {
    detections.push_back(make_tuple(ob_label, conf, xmin, ymin, xmax, ymax));
  }
}


class RenderObjectsOp : public OpKernel {
 public:
  explicit RenderObjectsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("out_channels", &out_channels_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& objects_label_tensor = context->input(0);
    auto objects_label = objects_label_tensor.flat<string>()(0);
    const Tensor& im_ht_tensor = context->input(1);
    auto im_ht = im_ht_tensor.flat<long long>()(0);
    const Tensor& im_wd_tensor = context->input(2);
    auto im_wd = im_wd_tensor.flat<long long>()(0);
    const Tensor& out_wd_tensor = context->input(3);
    auto out_wd = out_wd_tensor.flat<long long>()(0);
    int out_ht = ((im_ht * out_wd * 1.0) / im_wd);

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
    vector<tuple<int,float,float,float,float,float>> detections;
    read_detections(objects_label, detections);
    for (int i = 0; i < out_wd; i++) {
      for (int j = 0; j < out_ht; j++) {
        for (int k = 0; k < out_channels_; k++) {
          output(j, i, k) = 0;
        }
      }
    }

    if (out_channels_ != 3) {  // i.e. not doing a RGB output
      for (unsigned int i = 0; i < detections.size(); i++) {
        int xmin = get<2>(detections[i]) * out_wd;
        int ymin = get<3>(detections[i]) * out_ht;
        int xmax = get<4>(detections[i]) * out_wd;
        int ymax = get<5>(detections[i]) * out_ht;
        int ob_label = get<0>(detections[i]);
        float conf = get<1>(detections[i]);
        for (int c = max(0, (int) xmin); c < min(xmax, (int) out_wd); c++) {
          for (int r = max(0, (int) ymin); r < min(ymax, (int) out_ht); r++) {
            output(r, c, ob_label) = conf;
          }
        }
      }
    } else {
      cerr << "render_objects: unable to render RGB currently." << endl;
    }
  }

 private:
  int out_channels_;
};

REGISTER_KERNEL_BUILDER(Name("RenderObjects").Device(DEVICE_CPU), RenderObjectsOp);
