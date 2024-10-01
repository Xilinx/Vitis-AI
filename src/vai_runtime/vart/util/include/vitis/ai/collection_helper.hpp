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
#include <memory>
#include <sstream>
#include <vector>

namespace vitis {
namespace ai {
template <typename T1, typename F>
std::vector<decltype(std::declval<F>()(std::declval<const T1&>()))> vec_map(
    const std::vector<T1>& v, F&& f) {
  auto ret =
      std::vector<decltype(std::declval<F>()(std::declval<const T1&>()))>();
  ret.reserve(v.size());
  for (const auto& x : v) {
    ret.emplace_back(f(x));
  }
  return ret;
}

template <typename T>
std::vector<T*> vector_unique_ptr_get(
    const std::vector<std::unique_ptr<T>>& from) {
  return vec_map(from, [](const std::unique_ptr<T>& x) { return x.get(); });
}

template <typename T>
std::vector<const T*> vector_unique_ptr_get_const(
    const std::vector<std::unique_ptr<T>>& from) {
  return vec_map(from, [](const std::unique_ptr<T>& x) {
    return const_cast<const T*>(x.get());
  });
};

/*
template <typename T>
std::string to_string(const T& x) {
  std::ostringstream str;
  str << x;
  return str.str();
}
    */
}  // namespace ai
}  // namespace vitis
