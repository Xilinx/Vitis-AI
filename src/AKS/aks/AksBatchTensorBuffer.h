/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <vart/runner.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

namespace AKS {

class AksBatchTensorBuffer : public vart::TensorBuffer {
 public:
  /// Create buffers from a vector of tensors
  explicit AksBatchTensorBuffer(std::vector<std::unique_ptr<xir::Tensor>> tensors)
      : TensorBuffer{tensors.front().get()}, tensors_{std::move(tensors)} {
    buffers_.reserve(tensors.size());
    for (const auto& tensor : tensors_) {
      buffers_.emplace_back(tensor->get_data_size());
    }
  }

  explicit AksBatchTensorBuffer(std::vector<std::unique_ptr<xir::Tensor>> tensors,
                               std::vector<std::vector<char>> buffers)
      : TensorBuffer{tensors.front().get()}
      , buffers_{std::move(buffers)}
      , tensors_{std::move(tensors)} {}

  ~AksBatchTensorBuffer() override = default;

  const std::vector<const xir::Tensor*> get_tensors() const {
    std::vector<const xir::Tensor*> t;
    t.reserve(tensors_.size());
    for(const auto& tensor: tensors_) {
      t.push_back(tensor.get());
    }
    return t;
  }

  // Access each input in the batch via tb.data(0), tb.data(1) etc.
  std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) override {
    if (idx.empty()) {
      int elems = 0;
      for (const auto& tensor : tensors_) {
        elems += tensor->get_data_size();
      }
      return {reinterpret_cast<uint64_t>(buffers_.front().data()), elems};
    } else if (idx.size() == 1) {
      int batch = idx[0];
      int elems =
          std::accumulate(std::next(tensors_.begin(), batch), tensors_.end(), 0,
                          [](int acc, const auto& tensor) {
                            return acc + tensor->get_data_size();
                          });
      return {reinterpret_cast<uint64_t>(buffers_.at(batch).data()), elems};
    } else {
      // Access tensor at given 'batch' and compute stride for it
      const auto& tensor = tensors_.at(idx[0]);
      const auto& shape = tensor->get_shape();
      std::vector<int> stride(shape.size(), 1);
      std::partial_sum(shape.crbegin(), std::prev(shape.crend()),
                       std::next(stride.rbegin()), std::multiplies<>());

      // Compute the offset in this tensor
      int offset = 0;
      for (int i = 1; i < idx.size(); ++i) {
        offset += idx[i] * stride[i];
      }
      char* base = buffers_.at(idx[0]).data() + offset;

      // Now compute the "rest" size
      std::size_t rest = tensor->get_data_size() - offset;
      for (int i = idx[0] + 1; i < tensors_.size(); ++i) {
        rest += tensors_[i]->get_data_size();
      }

      return {reinterpret_cast<uint64_t>(base), rest};
    }
  }

 private:
  std::vector<std::vector<char>> buffers_;   /// Store each input in a batch
  std::vector<std::unique_ptr<xir::Tensor>> tensors_;  /// Store corresponding tensor
};

}  // namespace AKS
