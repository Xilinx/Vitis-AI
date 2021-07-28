/*
 * Copyright 2019 Xilinx Inc.
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

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"

#include <xrt.h>

#include "vart/runner.hpp"

namespace vart {
namespace assistant {
static uint64_t get_physical_address(const xclDeviceHandle& handle,
                                     const unsigned int bo) {
  xclBOProperties p;
  auto error_code = xclGetBOProperties(handle, bo, &p);
  uint64_t phy = 0u;
  if (error_code != 0) {
    LOG(INFO) << "cannot xclGetBOProperties !";
  }
  phy = error_code == 0 ? p.paddr : -1;
  return phy;
}

std::unique_ptr<vart::TensorBuffer> XrtBoTensorBuffer::create(
    vart::xrt_bo_t bo, const xir::Tensor* tensor) {
  return std::make_unique<XrtBoTensorBuffer>(bo, tensor);
}

XrtBoTensorBuffer::XrtBoTensorBuffer(vart::xrt_bo_t bo,
                                     const xir::Tensor* tensor)
    : TensorBuffer(tensor), bo_{bo} {
  CHECK(tensor->has_attr("reg_id")) << "tensor: " << tensor->to_string();
  CHECK(tensor->has_attr("ddr_addr")) << "tensor: " << tensor->to_string();
  CHECK(tensor->has_attr("location")) << "tensor: " << tensor->to_string();
  // auto reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  ddr_addr_ = (size_t)tensor->template get_attr<int>("ddr_addr");
  auto location = (size_t)tensor->template get_attr<int>("location");
  CHECK_EQ(location, 1);
  phy_addr_ = get_physical_address(bo.xrt_handle, bo.xrt_bo_handle);
  // TODO: assumue one bo one tensor, and the tensor should be on the
  // TODO: this is the bug for image bundling.
  size_ = tensor->get_data_size() / tensor->get_shape()[0];
}

std::pair<std::uint64_t, std::size_t> XrtBoTensorBuffer::data(
    const std::vector<std::int32_t> idx) {
  return std::make_pair((uint64_t)0u, (size_t)0u);
}
vart::TensorBuffer::location_t XrtBoTensorBuffer::get_location() const {
  // TODO: how to get device id from a XRT handle.
  return vart::TensorBuffer::location_t::DEVICE_0;
}
std::pair<uint64_t, size_t> XrtBoTensorBuffer::data_phy(
    const std::vector<std::int32_t> idx) {
  return std::make_pair((uint64_t)phy_addr_ + ddr_addr_, size_);
}
void XrtBoTensorBuffer::sync_for_read(uint64_t offset, size_t size) {
  // because it is not mapped, so that it is not so meaningful to  flush cache
  return;
}

void XrtBoTensorBuffer::sync_for_write(uint64_t offset, size_t size) {
  // because it is not mapped, so that it is not so meaningful to invalidate
  // cache
  return;
}

void XrtBoTensorBuffer::copy_from_host(size_t batch_idx, const void* buf,
                                       size_t size, size_t offset) {
  LOG(FATAL) << "TODO: not implemented yet";
}

void XrtBoTensorBuffer::copy_to_host(size_t batch_idx, void* buf, size_t size,
                                     size_t offset) {
  LOG(FATAL) << "TODO: not implemented yet";
}
}  // namespace assistant
}  // namespace vart
