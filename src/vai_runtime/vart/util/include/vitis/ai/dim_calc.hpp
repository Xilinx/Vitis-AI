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
#include <cstdlib>
#include <map>
#include <vector>
namespace vitis {
namespace ai {

class DimCalc {
 public:
  DimCalc() = delete;
  explicit DimCalc(const std::vector<size_t>& dims);
  explicit DimCalc(const std::vector<int32_t>& dims);
  explicit DimCalc(const std::vector<size_t>& dims,
                   const std::vector<size_t>& strides);
  ~DimCalc() = default;
  DimCalc(const DimCalc& other) = default;
  DimCalc& operator=(const DimCalc& rhs) = default;

 public:
  std::pair<std::vector<size_t>, size_t> next(
      const std::vector<size_t>& idx) const;

  size_t offset(const std::vector<size_t>& idx) const;
  size_t offset(const std::vector<int>& idx) const;
  std::vector<int> index(size_t offset) const;

 public:
  const std::vector<size_t> dims_;
  const std::vector<size_t> strides_;
  const int non_id_;
};

}  // namespace ai
}  // namespace vitis
