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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_OPS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {

class DatasetToGraphOp : public OpKernel {
 public:
  static constexpr const char* const kStatefulWhitelist = "stateful_whitelist";

  explicit DatasetToGraphOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  std::vector<string> whitelisted_stateful_ops_;
};

class DatasetCardinalityOp : public OpKernel {
 public:
  explicit DatasetCardinalityOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class DatasetFromGraphOp : public OpKernel {
 public:
  static constexpr const char* const kGraphDef = "graph_def";
  static constexpr const char* const kHandle = "handle";

  explicit DatasetFromGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_OPS_H_
