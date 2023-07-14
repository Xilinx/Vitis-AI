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
#include <memory>
#include <string>
#include <vart/runner.hpp>
namespace xir {
class Tensor;
class Attrs;
}  // namespace xir
namespace vart {
namespace assistant {
class TensorBufferAllocator {
 public:
  /** @brief create a tensor buffer allocator */
  static std::unique_ptr<TensorBufferAllocator> create(const xir::Attrs* attrs);
  explicit TensorBufferAllocator();

  TensorBufferAllocator(const TensorBufferAllocator&) = delete;
  TensorBufferAllocator& operator=(const TensorBufferAllocator& other) = delete;

  virtual ~TensorBufferAllocator() = default;

 public:
  /** @brief allocate a vector of tensor buffers.
   *  @param subgraph
   *  @param input_tensors
   *  @param output_tensors
   *  @return <input_tensor_buffer, output_tensor_buffers>
   *
   * */
  virtual std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
                    std::vector<std::unique_ptr<vart::TensorBuffer>>>
  allocate(const xir::Subgraph* subgraph,
           const std::vector<const xir::Tensor*>& input_tensors,
           const std::vector<const xir::Tensor*>& output_tensors) = 0;
};
}  // namespace assistant
}  // namespace vart
