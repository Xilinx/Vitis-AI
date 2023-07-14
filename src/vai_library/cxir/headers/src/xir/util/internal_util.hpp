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
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "UniLog/UniLog.hpp"
#include "xir/tensor/tensor.hpp"

#define HEAD_DELIMITER "("
#define TAIL_DELIMITER ")"

namespace xir {
namespace internal {
// const converter
// vector related
template <typename T>
std::vector<T*> cast_from_const_vector(const std::vector<const T*>& s) {
  auto ret = std::vector<T*>();
  ret.reserve(s.size());
  for (auto p : s) {
    ret.push_back(const_cast<T*>(p));
  }
  return ret;
}

template <typename T>
std::vector<const T*> cast_to_const_vector(const std::vector<T*>& s) {
  auto ret = std::vector<const T*>();
  ret.reserve(s.size());
  for (auto p : s) {
    ret.push_back(static_cast<const T*>(p));
  }
  return ret;
}

// set related
template <typename T>
std::set<T*> cast_from_const_set(const std::set<const T*>& s) {
  auto ret = std::set<T*>();
  for (auto p : s) {
    ret.insert(const_cast<T*>(p));
  }
  return ret;
}

template <typename T>
std::set<const T*> cast_to_const_set(const std::set<T*>& s) {
  auto ret = std::set<const T*>();
  for (auto p : s) {
    ret.insert(static_cast<const T*>(p));
  }
  return ret;
}

// op related
std::vector<const Op*> vec_input_ops(
    const std::map<std::string, std::vector<const Op*>>& input_ops);
std::vector<Op*> vec_input_ops(
    const std::map<std::string, std::vector<Op*>>& input_ops);

// round related
float dpu_round_float(const float& input);
float py3_round_float(const float& input);

// data_related
template <typename Dtype>
std::vector<char> streamize(const std::vector<Dtype>& ori) {
  std::vector<char> ret;
  auto vec_char_len = (sizeof(Dtype) / sizeof(char)) * ori.size();
  ret.reserve(vec_char_len);
  const char* char_ptr = reinterpret_cast<const char*>(ori.data());
  for (auto idx = 0U; idx < vec_char_len; idx++) {
    ret.push_back(char_ptr[idx]);
  }
  return ret;
}

template <typename Dtype>
std::vector<Dtype> restreamize(const std::vector<char>& ori) {
  std::vector<Dtype> ret;
  auto vec_dtype_len = ori.size() * sizeof(char) / sizeof(Dtype);
  ret.reserve(vec_dtype_len);
  const Dtype* dtype_ptr = reinterpret_cast<const Dtype*>(ori.data());
  for (auto idx = 0U; idx < vec_dtype_len; idx++) {
    ret.push_back(dtype_ptr[idx]);
  }
  return ret;
}

}  // namespace internal
}  // namespace xir
