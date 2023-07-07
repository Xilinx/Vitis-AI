#ifndef _NNDCT_INCLUDE_CPU_BFP_H_
#define _NNDCT_INCLUDE_CPU_BFP_H_

#include <torch/extension.h>

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out);

torch::Tensor& to_bfp_v2(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out);

#endif // _NNDCT_INCLUDE_CPU_BFP_H_
