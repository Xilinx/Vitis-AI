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
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>

#include "vart/tensor_buffer.hpp"
#include "vitis/ai/tensor_buffer.hpp"

namespace vitis {
namespace ai {
class TensorBufferAdaptor : public vart::TensorBuffer {
 public:
  TensorBufferAdaptor(vitis::ai::TensorBuffer* self);
  virtual ~TensorBufferAdaptor() = default;

 private:
  virtual std::pair<uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) override;

 private:
  vitis::ai::TensorBuffer* self_;
  std::unique_ptr<xir::Tensor> tensor_;
};
}  // namespace ai
}  // namespace vitis
