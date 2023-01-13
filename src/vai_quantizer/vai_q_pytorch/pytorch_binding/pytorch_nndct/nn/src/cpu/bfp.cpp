#include "../../include/bfp.h"

uint32_t GetExponent(float v) {
  // Get the biased exponent.
  uint32_t uint_v = *reinterpret_cast<uint32_t*>(&v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

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

torch::Tensor calculate_shared_exponent(torch::Tensor tensor,
                                        int64_t block_size) {
  TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous.");
  // TODO(yuwang): Support double dtype.
  TORCH_CHECK(tensor.dtype() == at::kFloat, "Only support float tensor.");
  TORCH_CHECK(tensor.numel() % block_size == 0,
      "The number of elements of tensor must be divisible by 'block_size'")

  const float* tensor_data = tensor.data_ptr<float>();

  uint32_t shared_exp = 0;
  torch::Tensor result = at::empty({tensor.numel() / block_size},
      tensor.options().dtype(at::kInt));
  int* result_data = result.data_ptr<int>();
  // Loop over bounding box to find shared exponent
  for (int64_t i = 0; i < tensor.numel(); i++) {
    if (i % block_size == 0) {
      shared_exp = 0;
    }
    uint32_t exp = GetExponent(tensor_data[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
    if ((i + 1) % block_size == 0) {
      // Output unbiased value.
      result_data[i / block_size] = shared_exp - 127;
    }
  }
  return result;
}

void _to_bfp_shift(const float* input,
             int64_t block_size,
             int64_t bit_width,
             float* output) {
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int64_t i = 0; i < block_size; i++) {
    uint32_t exp = GetExponent(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  uint32_t m_bits = bit_width - 8;
  for (int64_t i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponent(input[i]);
    if (exp == 0 || exp == 0xff) {
      output[i] = input[i];
    } else {
      uint32_t v = *reinterpret_cast<const uint32_t*>(&input[i]);
      uint32_t sign = v & 0x80000000;
      uint32_t mantissa = v & 0x007fffff;

      uint32_t exp_diff = shared_exp - exp;

      // Implied 1
      mantissa = mantissa + (1 << 23);
      // Adjust for shared exponent
      mantissa = mantissa >> exp_diff;
      // Shift down to target bit width + 1
      mantissa = mantissa >> (23 - m_bits);
      // Rounding (with overflow check)
      if (mantissa != ((1 << (m_bits + 1)) - 1)) {
        mantissa += 1;
      }
      // Shift away last bit
      mantissa = mantissa >> 1;
      uint32_t result = sign | shared_exp | mantissa;
      output[i] = *reinterpret_cast<float*>(&result);
    }
  }
}

void BFPKernel(const float* input,
               float* output,
               int n,
               int bit_width,
               int index,
               int stride) {
  //printf("n=%d, index=%d, stride=%d\n", n, index, stride);
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int i = index; i < n; i += stride) {
    uint32_t exp = GetExponent(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = std::pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = std::pow(2.0, shared_exp_value + 1) - scale;
  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponent(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }
}

void BFPKernelV2(const float* input,
                 float* output,
                 int offset,
                 int bit_width,
                 int block_size) {
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int i = 0; i < block_size; i++) {
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = std::pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = std::pow(2.0, shared_exp_value + 1) - scale;

  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }
}

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out) {
  CheckInput(tensor, bit_width, block_size);

  //torch::Tensor result = at::empty_like(tensor);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  int num_blocks = tensor.numel() / block_size;
  for (int index = 0; index < num_blocks; index++) {
    BFPKernel(input, output, tensor.numel(), bit_width, index, num_blocks);
  }
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

  //torch::Tensor result = at::empty_like(tensor);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  int num_blocks = tensor.numel() / block_size;
  for (int index = 0; index < num_blocks; index++) {
    BFPKernel(input, output, index * block_size + block_size, bit_width,
        index * block_size, 1);
  }
  return out;
}

//TORCH_LIBRARY_IMPL(vai, CPU, m) {
//  m.impl("to_bfp_v2", to_bfp_cpu)
//}
