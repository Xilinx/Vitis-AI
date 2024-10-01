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
#include <stdlib.h>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "fmt.hpp"

#define TRACE_VAR(x) #x, (x)

namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
	using std::vector;
#pragma pack(push, 1)
template <typename... Ts>
class trace_payload {
 public:
  trace_payload(Ts... args) : payload_(args...){};
  trace_payload(){};

  template <int N>
  inline std::enable_if_t<(N == 0), bool> get_(vector<std::string>& buf) {
    auto data = std::get<N>(payload_);
    buf.push_back(to_string(data));
    return true;
  }

  template <int N>
  inline std::enable_if_t<(N > 0), bool> get_(vector<std::string>& buf) {
    get_<N - 1>(buf);
    auto data = std::get<N>(payload_);
    buf.push_back(to_string(data));
    return true;
  }

  inline void to_vector(vector<std::string>& buf) {
    constexpr int payload_size = std::tuple_size<decltype(payload_)>::value - 1;
    get_<payload_size>(buf);
  }

  inline size_t size(void) { return size_; };

 private:
  std::tuple<Ts...> payload_;
  enum : uint16_t { size_ = sizeof(std::tuple<Ts...>) };
};

template <typename... Ts>
trace_payload<Ts...> make_payload(Ts... args) {
  return trace_payload<Ts...>(args...);
};
#pragma pack(pop)

}  // namespace vitis::ai::trace
