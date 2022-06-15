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

#include <cmath>
#include <string>
#include <memory>
#include <algorithm>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

namespace AKS {

class AksTensorBuffer: public vart::TensorBuffer {
  public:
    // Ctor
    explicit AksTensorBuffer(std::unique_ptr<xir::Tensor> tensor);
    // swap
    friend void swap(AksTensorBuffer& a, AksTensorBuffer& b) noexcept;
    // Copy Ctor
    AksTensorBuffer (const AksTensorBuffer& src);
    // Copy Assignment
    AksTensorBuffer & operator = (const AksTensorBuffer& src);
    // Move Ctor
    AksTensorBuffer (AksTensorBuffer&& src) noexcept;
    // Move Assignment 
    AksTensorBuffer & operator = (AksTensorBuffer&& src) noexcept;
    // data method
    std::pair<uint64_t, size_t> data(
        const std::vector<std::int32_t> idx = {}) override;
    // Dtor
    ~AksTensorBuffer () override = default;

  private:
    std::vector<char> buffer_;
    std::unique_ptr<xir::Tensor> tsor_;
};

}//namespace AKS
