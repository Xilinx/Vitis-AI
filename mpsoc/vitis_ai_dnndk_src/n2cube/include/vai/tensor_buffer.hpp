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

#include "vai/tensor.hpp"

namespace vitis {
namespace ai {

class TensorBuffer {
protected:
  explicit TensorBuffer(const Tensor *tensor);

public:
  virtual ~TensorBuffer() = default;

public:
  virtual std::pair<void *, std::size_t>
  data(const std::vector<std::int32_t> idx = {}) = 0;

public:
  const Tensor *get_tensor() const;

protected:
  const Tensor *tensor_;
};

class CpuFlatTensorBuffer : public TensorBuffer {
public:
  explicit CpuFlatTensorBuffer(void *data, const Tensor *tensor);
  virtual ~CpuFlatTensorBuffer() = default;

public:
  virtual std::pair<void *, size_t>
  data(const std::vector<int> idx = {}) override;

private:
  void *data_;
};
}
}
