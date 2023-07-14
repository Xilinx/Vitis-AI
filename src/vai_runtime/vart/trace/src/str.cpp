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

#include "str.hpp"

namespace vitis::ai::trace {

// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;

str_pool g_pool_;

size_t str_pool_size(void) { return pool_instance().size(); }

str_pool::str_pool() : index(0), pool{} {};
str_pool::~str_pool(){};

str_id str_pool::add_str(const char* str_) {
  if (pool[str_]) {
    return pool[str_];
  } else {
    pool[str_] = new_idx();
    return pool[str_];
  }
};

const char* str_pool::idx_to_str(str_id id) {
  for (const auto& [str, idx] : pool) {
    if (idx == id) {
      return str;
    }
  }
  return NULL;
};

size_t str_pool::size() { return pool.size(); };

str::str(const char* str_) { idx_ = pool_instance().add_str(str_); };

// str::str() {
//    idx_ = 0;
//};
//
const char* str::to_string(void) { return pool_instance().idx_to_str(idx_); };
std::string to_srting(str s_) { return std::string(s_.to_string()); };
}  // namespace vitis::ai::trace
