#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOutChannels")
  .Attr("T: {float32, float64, int32, int64}")
  .Input("to_zero: T")  // must be 4-dim images
  .Input("channels: bool")  // list of true/false, false=>zero out that channel
  .Output("zeroed: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

template <typename T>
class ZeroOutChannelsOp : public OpKernel {
 public:
  explicit ZeroOutChannelsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_tensor_channel = context->input(1);
    auto input = input_tensor.tensor<T, 4>();
    auto input_channel = input_tensor_channel.flat<bool>();

    assert(input_tensor.shape().dims() == 4);
    int num_channels = input_tensor.shape().dim_size(3);
    assert(num_channels == input_tensor_channel.shape().dim_size(0));
    Tensor *output = NULL;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->tensor<T, 4>();
    for (int i = 0; i < input_tensor.shape().dim_size(0); i++) {
      for (int j = 0; j < input_tensor.shape().dim_size(1); j++) {
        for (int k = 0; k < input_tensor.shape().dim_size(2); k++) {
          for (int l = 0; l < input_tensor.shape().dim_size(3); l++) {
            if (input_channel(l) == false) {
              output_flat(i, j, k, l) = 0;
            } else {
              output_flat(i, j, k, l) = input(i, j, k, l);
            }
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ZeroOutChannels")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    ZeroOutChannelsOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOutChannels")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutChannelsOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOutChannels")
    .Device(DEVICE_CPU)
    .TypeConstraint<int>("T"),
    ZeroOutChannelsOp<int>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOutChannels")
    .Device(DEVICE_CPU)
    .TypeConstraint<long long>("T"),
    ZeroOutChannelsOp<long long>);
