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
#include "./tensor_buffer_adaptor.hpp"

#include "./convert_tensor.hpp"

namespace vitis {
namespace ai {

TensorBufferAdaptor::TensorBufferAdaptor(vitis::ai::TensorBuffer* self)
    : vart::TensorBuffer(convert_tensor(self->get_tensor()).release()),
      self_{self},
      tensor_{std::unique_ptr<xir::Tensor>(
          const_cast<xir::Tensor*>(get_tensor()))} {
  (void)convert_tensors({});  // supress warning
}

std::pair<uint64_t, std::size_t> TensorBufferAdaptor::data(
    const std::vector<std::int32_t> idx) {
  void* data;
  std::size_t size;
  std::tie(data, size) = self_->data(idx);
  return std::make_pair((uint64_t)data, size);
}
}  // namespace ai
}  // namespace vitis
