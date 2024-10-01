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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ringbuf.hpp"
#include "vaitrace_dbg.hpp"

namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
extern bool is_enabled(void);

template <typename TrFun, typename T>
class tevent : public traceEventBase {
 public:
  tevent(TrFun& tf, T& pl)
      : traceEventBase(sizeof(pl)), translate_f(tf), payload(pl) {}

  ~tevent() {}

  trace_entry_t get() {
    auto trace_entry = traceEventBase::get();

    vector<string> payload_buf;
    payload.to_vector(payload_buf);

    auto payload_map = translate_f(payload_buf);

    trace_entry.insert(payload_map.begin(), payload_map.end());

    return trace_entry;
  }

 private:
  TrFun& translate_f;
  T payload;
};

void push_info(trace_entry_t i);
class traceClass {
 public:
  traceClass(const char* name_, vector<string> items);
  ~traceClass() = default;

  template <typename... Ts>
  inline void add_trace(Ts... args) {
    if (!is_enabled()) return;
    auto payload = vitis::ai::trace::trace_payload<Ts...>(args...);

    auto e = new tevent<decltype(this->translate_f), decltype(payload)>(
        this->translate_f, payload);
    get_rbuf()->push(e);
  };

  template <typename... Ts>
  inline void add_info(Ts... args) {
    if (!is_enabled()) return;

    // Should stop compiling here if sizeof...(args) % 2 != 0
    static_assert(sizeof...(args) % 2 == 0);

    auto info = vitis::ai::trace::trace_payload<Ts...>(args...);

    trace_entry_t ret;

    ret.insert(make_pair("classname", this->classname));
    vector<string> buf;

    info.to_vector(buf);

    for (size_t i = 0; i < (buf.size() / 2); i++) {
      auto pos = 2 * i;
      ret.insert(make_pair(buf[pos], buf[pos + 1]));
    }
    // get_infobase()->push_back(ret);
    push_info(ret);
  };

  std::function<trace_entry_t(std::vector<std::string>)> translate_f =
      [this](std::vector<std::string> data) {
        trace_entry_t ret;

        ret.insert(std::make_pair("classname", this->classname));
        uint32_t data_size = (uint32_t)data.size();
        // MSVC NOTE: must use std::min<uint32_t> instead of std::min, otherwise
        // strange compilation error illegal token.
        auto col_num = std::min<uint32_t>(data_size, column_num);

        for (size_t i = 0; i < col_num; i++) {
          ret.insert(make_pair(this->column_names[i], data[i]));
        }
        return ret;
      };
  string classname;

 private:
  uint32_t column_num;
  vector<string> column_names;
};

traceClass* new_traceclass(const char* name_, vector<string> items);
traceClass* find_traceclass(const char* classname);

}  // namespace vitis::ai::trace
