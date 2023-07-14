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
#include "../include/vitis/ai/dim_calc.hpp"

#include <glog/logging.h>

#include "../include/vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DIM_CALC, "0");
namespace vitis {
namespace ai {

static std::vector<size_t> default_strides(const std::vector<size_t>& dims) {
  auto ret = std::vector<size_t>(dims.size());
  auto idx = dims.size();
  size_t acc = 1u;
  for (auto i = 0u; i < dims.size(); ++i) {
    idx = idx - 1;
    ret[idx] = acc;
    acc = dims[idx] * acc;
  }
  return ret;
}

static int find_first_non_continuous_dim(const std::vector<size_t>& dims,
                                         const std::vector<size_t>& strides) {
  CHECK(!strides.empty());
  CHECK_EQ(strides.size(), dims.size());
  auto ret = (signed)(strides.size()) - 1;
  size_t acc = 1u;
  while (ret >= 0 && (strides[ret] == acc)) {
    acc = dims[ret] * acc;
    ret = ret - 1;
  }
  return ret;
}

static size_t size_from(const std::vector<size_t>& dims, size_t from) {
  size_t ret = 1;
  for (; from < dims.size(); ++from) {
    ret = ret * dims[from];
  }
  return ret;
}

static size_t to_linear(const std::vector<size_t>& index,
                        const std::vector<size_t>& dims) {
  size_t ret = 0;
  CHECK_EQ(index.size(), dims.size());
  CHECK(!index.empty());
  for (auto i = 0u; i < index.size(); ++i) {
    ret = ret + index[i] * size_from(dims, i + 1);
  }
  return ret;
}

static std::vector<size_t> from_linear(size_t linear,
                                       const std::vector<size_t>& dims) {
  auto ret = std::vector<size_t>{};
  ret.reserve(dims.size());
  for (auto i = 0u; i < dims.size(); ++i) {
    auto size = size_from(dims, i + 1);
    ret.emplace_back(linear / size);
    linear = linear - ret.back() * size;
  }
  CHECK_EQ(linear, 0U);
  return ret;
}

static bool in_range(const std::vector<size_t>& index,
                     const std::vector<size_t>& dims) {
  return to_linear(index, dims) < size_from(dims, 0);
}

// static std::vector<size_t> next_index(const std::vector<size_t>& index,
//                                       const std::vector<size_t>& dims,
//                                       size_t size) {
//   return from_linear(to_linear(index, dims) + size, dims);
// }

DimCalc::DimCalc(const std::vector<size_t>& dims)
    : dims_{dims},
      strides_{default_strides(dims)},
      non_id_{find_first_non_continuous_dim(dims_, strides_)} {}

template <typename T>
static std::vector<size_t> to_size_t_vector(const std::vector<T>& x) {
  auto ret = std::vector<size_t>(x.size(), 0u);
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = (size_t)x[i];
  }
  return ret;
}

DimCalc::DimCalc(const std::vector<int32_t>& dims)
    : dims_{to_size_t_vector(dims)},
      strides_{default_strides(dims_)},
      non_id_{find_first_non_continuous_dim(dims_, strides_)} {}

DimCalc::DimCalc(const std::vector<size_t>& dims,
                 const std::vector<size_t>& strides)
    : dims_{dims},
      strides_{strides},
      non_id_{find_first_non_continuous_dim(dims_, strides_)} {}

static std::ostream& operator<<(std::ostream& s, const std::vector<size_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}

std::pair<std::vector<size_t>, size_t> DimCalc::next(
    const std::vector<size_t>& idx) const {
  if (!in_range(idx, dims_)) {
    return std::make_pair(idx, 0u);
  }
  if (non_id_ == -1) {
    auto sz = strides_[0] * dims_[0];
    auto next_idx = dims_;
    std::fill(next_idx.begin() + 1, next_idx.end(), 0u);
    auto left_sz = sz - offset(idx);
    return std::make_pair(next_idx, left_sz);
  }
  auto next_idx = idx;
  auto begin = next_idx.begin() + non_id_;
  std::fill(begin + 1, next_idx.end(), 0u);
  auto base_idx = next_idx;
  *begin = *begin + 1;
  next_idx = from_linear(to_linear(next_idx, dims_), dims_);
  auto off_base = offset(base_idx);
  auto sz = strides_[non_id_ + 1] * dims_[non_id_ + 1];
  auto left_sz = sz - (offset(idx) - off_base);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DIM_CALC))
      << "idx " << idx << " "
      << "base_idx " << base_idx << " "        //
      << "next_idx " << next_idx << " "        //
      << "dims_ " << dims_ << " "              //
      << "strides_ " << strides_ << " "        //
      << "sz " << sz << " "                    //
      << "left_sz " << left_sz << " "          //
      << "off_base " << off_base << " "        //
      << "offset(idx) " << offset(idx) << " "  //
      ;
  return std::make_pair(next_idx, left_sz);
}

size_t DimCalc::offset(const std::vector<size_t>& idx) const {
  size_t ret = 0u;
  CHECK_EQ(idx.size(), strides_.size());
  for (auto i = 0u; i < idx.size(); ++i) {
    ret = ret + idx[i] * strides_[i];
  }
  return ret;
}
size_t DimCalc::offset(const std::vector<int>& idx) const {
  return offset(to_size_t_vector(idx));
}

std::vector<int> DimCalc::index(size_t offset) const {
  auto x = from_linear(offset, dims_);
  auto ret = std::vector<int>(x.size());
  for (auto i = 0u; i < x.size(); ++i) {
    ret[i] = (int)x[i];
  }
  return ret;
}
}  // namespace ai
}  // namespace vitis
