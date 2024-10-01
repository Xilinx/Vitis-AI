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
#include <assert.h>
#include <stdlib.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using str_id = uint16_t;

namespace vitis {
namespace ai {
namespace trace {

class str_pool {
 public:
  str_pool();
  ~str_pool();
  str_id add_str(const char* str_);
  const char* idx_to_str(str_id id);
  size_t size();

 private:
  str_id new_idx() {
    index++;
    return index;
  }
  str_id index;
  std::map<const char*, str_id> pool;
};

extern str_pool g_pool_;

inline str_pool& pool_instance() { return g_pool_; };

class str {
 public:
  str(const char* str_);
  const char* to_string(void);

 private:
  str_id idx_;
};

size_t str_pool_size(void);

std::string to_srting(str s_);

}  // namespace trace
}  // namespace ai
}  // namespace vitis
