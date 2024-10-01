/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "vart/tensor_buffer.hpp"

#include <xir/tensor/tensor.hpp>

namespace vart {
namespace mm {

class HostFlatTensorBuffer : public TensorBuffer {
 public:
  explicit HostFlatTensorBuffer(const xir::Tensor* tensor);
  explicit HostFlatTensorBuffer(const xir::Tensor* tensor,
                                std::vector<int32_t> strides);
  virtual ~HostFlatTensorBuffer();

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;

 public:
  const xir::DataType data_type;
  const std::vector<int32_t> shape;    // element
  const std::vector<int32_t> strides;  // bit
  const int32_t last_continued_dim;

 private:
  char* data_;
};

void init_from_file(HostFlatTensorBuffer* buffer, std::string file_name);
void dump_to_file(HostFlatTensorBuffer* buffer, std::string file_name);

std::vector<int32_t> get_strides(const xir::Tensor* tensor,
                                 bool ignore_def = false);

void tensorbuffer_copy(HostFlatTensorBuffer* buffer_src,
                       HostFlatTensorBuffer* buffer_dest);

std::pair<std::unique_ptr<HostFlatTensorBuffer>, std::unique_ptr<xir::Tensor>>
transform_to_fix_buffer(TensorBuffer* buffer, int32_t fix_point,
                        int32_t bit_width, bool if_signed,
                        std::string round_mode);
void transform_to_fix_buffer(TensorBuffer* buffer_src,
                             TensorBuffer* buffer_dest, std::string round_mode);

}  // namespace mm
}  // namespace vart
