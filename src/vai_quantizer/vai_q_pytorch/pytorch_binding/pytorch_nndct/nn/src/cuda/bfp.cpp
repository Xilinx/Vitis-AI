#include "../../include/bfp.h"
#include "../../../../../include/cuda/bfp_kernel.h"

void CheckInput(const torch::Tensor& tensor,
                int64_t bit_width,
                int64_t block_size) {
  TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous.");
  // TODO(yuwang): Support double dtype.
  TORCH_CHECK(tensor.dtype() == at::kFloat,
    "Tensor with dtype float32 can be quantized to BFP, but got ",
    tensor.toString());
  TORCH_CHECK(tensor.numel() % block_size == 0,
      "The number of elements of tensor must be divisible by 'block_size'");
  TORCH_CHECK(bit_width >= 10 && bit_width <= 16,
      "Bitwidth must be in [10, 16]");
}

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out) {
  CheckInput(tensor, bit_width, block_size);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  LaunchBFPKernel(input, output, tensor.numel(), bit_width, block_size);
  return out;
}

torch::Tensor& to_bfp_v2(const torch::Tensor& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         torch::Tensor& out) {
  CheckInput(tensor, bit_width, block_size);
  int axis_size = tensor.size(tensor.dim() - 1);
  TORCH_CHECK(axis_size % block_size == 0,
      "The number of elements in last axis must be divisible by 'block_size'");

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();
  LaunchBFPKernelV2(
      input, output, tensor.numel(), axis_size, bit_width, block_size);
  return out;
}

//TORCH_LIBRARY_IMPL(vai, CUDA, m) {
//  m.impl("to_bfp_v2", to_bfp_v2);
//}
