/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/einsum_op.h"

namespace tensorflow {

#define DECLARE_GPU_SPECS_NDIM(T, NDIM)                              \
  template struct functor::StrideFunctor<Eigen::GpuDevice, T, NDIM>; \
  template struct functor::InflateFunctor<Eigen::GpuDevice, T, NDIM>;

#define DECLARE_GPU_SPECS(T)    \
  DECLARE_GPU_SPECS_NDIM(T, 1); \
  DECLARE_GPU_SPECS_NDIM(T, 2); \
  DECLARE_GPU_SPECS_NDIM(T, 3); \
  DECLARE_GPU_SPECS_NDIM(T, 4); \
  DECLARE_GPU_SPECS_NDIM(T, 5); \
  DECLARE_GPU_SPECS_NDIM(T, 6);

TF_CALL_float(DECLARE_GPU_SPECS);
TF_CALL_double(DECLARE_GPU_SPECS);
TF_CALL_complex64(DECLARE_GPU_SPECS);
TF_CALL_complex128(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS_NDIM
#undef DECLARE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
