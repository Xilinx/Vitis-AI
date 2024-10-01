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
#include "vart/tensor_buffer_unowned_device.hpp"

#include <xir/tensor/tensor.hpp>

#include "vart/tensor_mirror_attrs.hpp"
#include "vitis/ai/dim_calc.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_TB_UNOWNED_DEVICE, "0")

namespace vart {
static std::vector<int32_t> change_shape(const xir::Tensor* tensor,
                                         size_t new_batch) {
  auto shape = tensor->get_shape();
  shape[0] = new_batch;
  return shape;
}

TensorBufferUnownedDevice::TensorBufferUnownedDevice(const xir::Tensor* tensor,
                                                     uint64_t batch_addr[],
                                                     size_t batch_size)
    : TensorBuffer(
          vart::TensorMirrorAttrs::create(
              tensor, change_shape(tensor, batch_size), tensor->get_data_type())
              .release()),
      tensor_(const_cast<xir::Tensor*>(get_tensor())) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TB_UNOWNED_DEVICE))
      << "TensorBufferUnownedDevice "
      << "@" << (void*)this << " created";
  host_phy_addr_.reserve(batch_size);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TB_UNOWNED_DEVICE))
      << " init unowned device tensor buffer with batch " << batch_size;
  for (size_t idx = 0; idx < batch_size; idx++) {
    host_phy_addr_.emplace_back(batch_addr[idx]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_TB_UNOWNED_DEVICE))
        << "device addr[" << idx << "] = 0x" << std::hex << batch_addr[idx];
  }
}

TensorBuffer::location_t TensorBufferUnownedDevice::get_location() const {
  // return TensorBuffer::location_t::HOST_PHY;
  return TensorBuffer::location_t::DEVICE_0;
}

std::pair<uint64_t, size_t> TensorBufferUnownedDevice::data_phy(
    const std::vector<std::int32_t> idx) {
  auto dims = get_tensor()->get_shape();
  auto batch_size = dims[0];
  dims[0] = 1;
  auto calc = vitis::ai::DimCalc(dims);

  auto indim = std::vector<int32_t>(idx);
  auto batch_idx = indim[0];
  indim[0] = 0;
  auto offset = calc.offset(indim);
  auto phy_addr = host_phy_addr_[batch_idx] + offset;
  auto size = get_tensor()->get_data_size() / batch_size - offset;

  return std::make_pair(phy_addr, size);
}

std::pair<std::uint64_t, std::size_t> TensorBufferUnownedDevice::data(
    const std::vector<std::int32_t> idx) {
  return std::make_pair(0u, 0u);
}

}  // namespace vart
