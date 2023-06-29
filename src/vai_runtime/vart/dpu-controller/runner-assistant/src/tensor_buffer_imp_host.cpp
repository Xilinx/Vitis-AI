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

#include "./tensor_buffer_imp_host.hpp"

#include <sstream>
#include <xir/tensor/tensor.hpp>

#include "vitis/ai/dim_calc.hpp"
namespace vart {
namespace dpu {
TensorBufferExtImpHost::TensorBufferExtImpHost(const xir::Tensor* tensor)
    : TensorBuffer(tensor), buffer_( (size_t)tensor->get_data_size()) {}

vart::TensorBuffer::location_t TensorBufferExtImpHost::get_location() const {
  return location_t::HOST_VIRT;
}

std::pair<uint64_t, size_t> TensorBufferExtImpHost::data_phy(
    const std::vector<std::int32_t> idx) {
  LOG(FATAL) << "not available: this=" << to_string();
  return std::make_pair((uint64_t)0u, (size_t)0u);
}

std::pair<std::uint64_t, std::size_t> TensorBufferExtImpHost::data(
    const std::vector<std::int32_t> idx) {
  auto size_of_elt = get_tensor()->get_data_type().bit_width / 8;
  auto calc = vitis::ai::DimCalc(get_tensor()->get_shape());
  auto offset = calc.offset(idx);
  auto size_in_elt = buffer_.size() / size_of_elt - offset;
  auto ret = (uint64_t)&buffer_[offset * size_of_elt];
  return std::make_pair(ret, size_in_elt * size_of_elt);
}
void TensorBufferExtImpHost::sync_for_read(uint64_t offset, size_t size) {
  // noop;
}

void TensorBufferExtImpHost::sync_for_write(uint64_t offset, size_t size) {
  // noop;
}
// void TensorBufferExtImpHost::copy_from_host(const void* buf, size_t size,
//                                             size_t offset) {
//   LOG(FATAL) << "TODO";
// }
// void TensorBufferExtImpHost::copy_to_host(void* buf, size_t size,
//                                           size_t offset) {
//   LOG(FATAL) << "TODO";
// }

}  // namespace dpu
}  // namespace vart
