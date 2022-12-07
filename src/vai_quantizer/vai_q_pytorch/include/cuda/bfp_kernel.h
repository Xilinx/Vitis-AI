#ifndef _INCLUDE_CUDA_BFP_KERNEL_H_
#define _INCLUDE_CUDA_BFP_KERNEL_H_

void LaunchBFPKernel(const float* input,
                     float* output,
                     int n,
                     int bit_width,
                     int block_size);

void LaunchBFPKernelV2(const float* input,
                       float* output,
                       const int n,
                       const int axis_size,
                       const int bit_width,
                       const int block_size);

#endif // _INCLUDE_CUDA_BFP_KERNEL_H_
