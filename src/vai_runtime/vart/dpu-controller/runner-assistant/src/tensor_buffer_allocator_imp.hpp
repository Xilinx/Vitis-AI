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

#include "vart/assistant/tensor_buffer_allocator.hpp"

namespace {
class TensorBufferAllocatorImp : public vart::assistant::TensorBufferAllocator {
 public:
  explicit TensorBufferAllocatorImp(const xir::Attrs* attrs);

  TensorBufferAllocatorImp(const TensorBufferAllocatorImp&) = delete;
  TensorBufferAllocatorImp& operator=(const TensorBufferAllocatorImp& other) =
      delete;

  virtual ~TensorBufferAllocatorImp();

 private:
  /** @brief allocate a vector of buffers.
   * */
  virtual std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
                    std::vector<std::unique_ptr<vart::TensorBuffer>>>
  allocate(const xir::Subgraph* subgraph,
           const std::vector<const xir::Tensor*>& input_tensors,
           const std::vector<const xir::Tensor*>& output_tensors) override;

 private:
  const xir::Attrs* attrs_;
};
}  // namespace
