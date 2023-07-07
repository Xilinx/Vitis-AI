#include "../../include/cuda/bfp_kernel.h"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

__device__ uint32_t GetExponent(float v) {
  // Get the biased exponent.
  uint32_t uint_v = *reinterpret_cast<uint32_t*>(&v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

__global__ void BFPKernel(int bit_width,
                          int n,
                          const float* input,
                          float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  uint32_t shared_exp = 0;
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
    auto exp = GetExponent(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = max(-max_v, min(x, max_v));
    }
  }
}

void LaunchBFPKernel(const float* input,
                     float* output,
                     int n,
                     int bit_width,
                     int block_size) {
  BFPKernel<<<n / block_size, 1>>>(bit_width, n, input, output);
}

__global__ void BFPKernelV2(const float* input,
                            float* output,
                            const int num_threads,
                            const int axis_size,
                            const int bit_width,
                            const int block_size) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {            
    return;
  }

  const int axis_blocks = axis_size / block_size;
  const int block_index = index % axis_blocks;
  const int axis_index = index / axis_blocks;
  //if (threadIdx.x >= 1) {
  //  printf("index=%d, axis_blocks=%d, block_index=%d, axis_index=%d\n",
  //      index, axis_blocks, block_index, axis_index);
  //}

  int offset = axis_index * axis_size + block_index * block_size;
  // Loop over bounding box to find shared exponent
  uint32_t shared_exp = 0;
  for (int i = 0; i < block_size; i++) {
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = pow(2.0, shared_exp_value + 1) - scale;
  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    auto index = offset + i;
    auto exp = GetExponent(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } else {
      // Round half to even.
      auto x = nearbyintf(input[index] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[index] = max(-max_v, min(x, max_v));
    }
  }
}

void LaunchBFPKernelV2(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int bit_width,
    const int block_size) {
  
  const int threads = n / block_size;
  
  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));
  BFPKernelV2<<<blocks, threads_per_block>>>(
    input,
    output,
    threads,
    axis_size,
    bit_width,
    block_size
  );
}
