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

#include "./tensor_buffer_imp_view.hpp"

#include <UniLog/UniLog.hpp>
#include <sstream>
#include <xir/tensor/tensor.hpp>

#include "vitis/ai/dim_calc.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR, "0")
namespace vart {
namespace dpu {
TensorBufferExtImpView::TensorBufferExtImpView(
    const xir::Tensor* tensor, size_t offset,
    std::shared_ptr<vart::TensorBuffer> backstore)
    : TensorBufferExt(xir::Tensor::clone(tensor).release()),
      tensor_{
          std::unique_ptr<xir::Tensor>(const_cast<xir::Tensor*>(get_tensor()))},
      offset_{offset},
      backstore_{backstore} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR) >= 3)
      << " TensorBufferExtImpView created: " << to_string();
  ;
}

TensorBufferExtImpView::~TensorBufferExtImpView() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR) >= 3)
      << " TensorBufferExtImpView destroyed: " << to_string();
  ;
}

TensorBuffer::location_t TensorBufferExtImpView::get_location() const {
  return backstore_->get_location();
}

std::pair<uint64_t, size_t> TensorBufferExtImpView::data_x(
    const std::vector<std::int32_t> idx_orig, int phy) {
  auto idx = std::vector<int32_t>(idx_orig);
  auto dims = get_tensor()->get_shape();
  auto batch_idx = idx[0];
  const auto batch = dims[0];
  idx[0] = 0;
  dims[0] = 1;
  auto calc1 = vitis::ai::DimCalc(dims);
  auto offset_in_single_batch = (int)calc1.offset(idx);
  auto size_in_single_batch = get_tensor()->get_data_size() / batch;
  UNI_LOG_CHECK((int)offset_in_single_batch <= (int)size_in_single_batch, VART_TENSOR_INFO_ERROR);
  auto size_left_in_single_batch =
      size_in_single_batch - offset_in_single_batch;
  UNI_LOG_CHECK(size_in_single_batch >= 0, VART_TENSOR_INFO_ERROR);

  uint64_t data_back;
  size_t size_back;
  if (phy) {
    std::tie(data_back, size_back) =
        backstore_->data_phy({batch_idx, offset_in_single_batch});
  } else {
    std::tie(data_back, size_back) =
        backstore_->data({batch_idx, offset_in_single_batch});
  }
  return std::make_pair(data_back + offset_, (size_t)size_left_in_single_batch);
}
std::pair<uint64_t, size_t> TensorBufferExtImpView::data_phy(
    const std::vector<std::int32_t> idx) {
  return data_x(idx, 1);
}
std::pair<std::uint64_t, std::size_t> TensorBufferExtImpView::data(
    const std::vector<std::int32_t> idx) {
  return data_x(idx, 0);
}

void TensorBufferExtImpView::sync_for_read(uint64_t offset, size_t size) {
  return backstore_->sync_for_read(offset + offset_, size);
}

void TensorBufferExtImpView::sync_for_write(uint64_t offset, size_t size) {
  return backstore_->sync_for_write(offset + offset_, size);
}

void TensorBufferExtImpView::copy_from_host(size_t batch_idx, const void* buf,
                                            size_t size, size_t offset) {
  return backstore_->copy_from_host(batch_idx, buf, size, offset + offset_);
}
void TensorBufferExtImpView::copy_to_host(size_t batch_idx, void* buf,
                                          size_t size, size_t offset) {
  return backstore_->copy_to_host(batch_idx, buf, size, offset + offset_);
}

XclBo TensorBufferExtImpView::get_xcl_bo(int batch_index) const {
  auto ret = XclBo{nullptr, 0u};
  auto p = dynamic_cast<vart::TensorBufferExt*>(backstore_.get());
  if (p) {
    ret = p->get_xcl_bo(batch_index);
  }
  return ret;
}
}  // namespace dpu
}  // namespace vart
