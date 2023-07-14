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
#include <map>
#include <vector>
#include <xir/op/op_def.hpp>

#include "vart/runner.hpp"
#include "vitis/ai/dpu_runner.hpp"
std::string to_string(
    const std::vector<vitis::ai::TensorBuffer*>& tensor_buffers);
std::string to_string(const std::vector<vart::TensorBuffer*>& tensor_buffers);
// std::string to_string(const std::vector<vart::TensorBuffer*>&
// tensor_buffers);
std::string to_string(const std::vector<xir::Tensor*>& tensors);
std::string to_string(
    const std::vector<const vitis::ai::TensorBuffer*>& tensor_buffers);
std::string to_string(
    const std::vector<const vart::TensorBuffer*>& tensor_buffers);
std::string to_string(const std::vector<const xir::Tensor*>& tensors);
std::string to_string(const vitis::ai::TensorBuffer* tensor_buffers);
std::string to_string(const vart::TensorBuffer* tensor_buffers);
std::string to_string(const xir::Tensor* tensors);
std::string to_string(const vitis::ai::Tensor* tensors);

namespace vitis {
namespace ai {
std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>
alloc_cpu_flat_tensor_buffers(const std::vector<vitis::ai::Tensor*>& tensors);
}  // namespace ai
}  // namespace vitis

namespace vart {
std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);
std::vector<std::unique_ptr<vart::TensorBuffer>> alloc_cpu_flat_tensor_buffers(
    const std::vector<const xir::Tensor*>& tensors);
std::unique_ptr<vart::TensorBuffer> alloc_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor);
typedef struct {
  void* data;
  size_t size;
} tensor_buffer_data_t;

tensor_buffer_data_t get_tensor_buffer_data(vart::TensorBuffer* tensor_buffer,
                                            size_t batch_index);
tensor_buffer_data_t get_tensor_buffer_data(vart::TensorBuffer* tensor_buffer,
                                            const std::vector<int>& idx);
void dump_tensor_buffer(const std::string& dir,
                        vart::TensorBuffer* tensor_buffer, int batch_base = 0);
}  // namespace vart

namespace xir {

std::string to_string(const xir::Attrs* attr);
std::string to_string(const xir::OpDef* opdef);
std::string to_string(const std::vector<OpArgDef>& argdef);
std::string to_string(const OpArgDef& argdef);

}  // namespace xir
