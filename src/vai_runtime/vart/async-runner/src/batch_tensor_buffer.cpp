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

#include "./batch_tensor_buffer.hpp"

#include "vart/runner.hpp"
namespace vart {
static std::unique_ptr<xir::Tensor> create_tensor(
    const std::vector<vart::TensorBuffer*>& from) {
  auto ret = std::unique_ptr<xir::Tensor>();
  auto batch = 0;
  auto batch_dims = std::vector<int32_t>();
  auto name = std::string();
  auto data_type = xir::DataType{};
  for (auto& b : from) {
    auto t = b->get_tensor();
    CHECK(t != nullptr) << "cannot get tensor from the tensor buffer";
    auto dims = t->get_shape();
    CHECK(!dims.empty()) << "dims.size() " << dims.size();
    batch = batch + dims[0];
    if (batch_dims.empty()) {
      batch_dims = dims;
      name = t->get_name();
      data_type = t->get_data_type();
    } else {
      CHECK_EQ(batch_dims.size(), dims.size());
      for (auto i = 1u; i < batch_dims.size(); ++i) {
        CHECK_EQ(batch_dims[i], dims[i])
            << " all tensor buffer should have same shepe except the batch, "
               "i.e. the first dimension. i = "
            << i;
      }
      CHECK_EQ(name, t->get_name()) << "all tensor should have same name";
      CHECK_EQ((int)data_type.type, (int)t->get_data_type().type)
          << "all tensor should have data_type";
      CHECK_EQ(data_type.bit_width, t->get_data_type().bit_width)
          << "all tensor should have bit_width";
    }
  }
  batch_dims[0] = batch;
  ret = xir::Tensor::create(name, batch_dims, data_type);
  return ret;
}

BatchTensorBuffer::BatchTensorBuffer(
    const std::vector<vart::TensorBuffer*>& tensor_buffers)
    : vart::TensorBuffer(create_tensor(tensor_buffers).release()),
      tensor_buffers_(tensor_buffers),
      // capture the tensor again,
      tensor_(const_cast<xir::Tensor*>(get_tensor())) {
  CHECK(!tensor_buffers_.empty());
}

BatchTensorBuffer ::~BatchTensorBuffer() {}

// static std::string to_string(int i) { return std::to_string(i); }
// template <typename T>
// std::string to_string(T begin, T end, char s = '[', char e = ']',
//                       char sep = ',') {
//   std::ostringstream str;
//   str << s;
//   int c = 0;
//   for (auto it = begin; it != end; ++it) {
//     if (c++ != 0) {
//       str << sep;
//     };
//     str << to_string(*it);
//   }
//   str << e;
//   return str.str();
// }

std::pair<uint64_t, size_t> BatchTensorBuffer::data(
    const std::vector<int> idx) {
  if (idx.empty()) {
    return tensor_buffers_[0]->data(idx);
  }
  size_t tb_idx = 0u;
  int batch = 0;
  //  Ddebug idx=[1,0,0,0] tb_idx=0 tensor_buffers_.size()=3 idx2=[1,0,0,0]
  //  batch=0
  auto dims = tensor_buffers_[tb_idx]->get_tensor()->get_shape();
  for (tb_idx = 0; tb_idx < tensor_buffers_.size() && idx[0] > batch;
       tb_idx++) {
    batch = batch + dims[0];
  }
  if (tb_idx >= tensor_buffers_.size()) {
    return std::make_pair(0u, 0u);
  }
  auto idx2 = idx;
  idx2[0] = idx[0] - batch;
  // LOG(INFO) << "Ddebug idx=" << to_string(idx.begin(), idx.end())
  //           << " tb_idx < tensor_buffers_.size()="
  //           << (tb_idx < tensor_buffers_.size()) << "  idx[0] < batch "
  //           << (idx[0] < batch) << " tb_idx=" << tb_idx  //
  //           << " " << idx[0] << "<" << batch             //
  //           << " tensor_buffers_.size()=" << tensor_buffers_.size()
  //           << " idx2=" << to_string(idx2.begin(), idx2.end())
  //           << " batch=" << batch;
  return tensor_buffers_[tb_idx]->data(idx2);
}
}  // namespace vart
