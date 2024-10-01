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

#include <xir/graph/graph.hpp>

#include "common.hpp"
#include "traceclass.hpp"

#ifdef ENABLE_XRT_TIMESTAMP
namespace xrt_core {
unsigned long time_ns();
}
#endif

namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
enum state {  func_end = 0,func_start = 1, marker };

void push_info(trace_entry_t i);

template <typename... Ts>
inline void info(Ts... args) {
  auto info = vitis::ai::trace::trace_payload<Ts...>(args...);

  trace_entry_t ret;

  vector<string> buf;

  info.to_vector(buf);

  for (size_t i = 0; i < buf.size(); i += 2) {
    ret.insert(make_pair(buf[i], buf[i + 1]));
  }
  push_info(ret);
};

// void start();
// void stop();
bool is_enabled();

void lock(void);
void lock(size_t &core_idx);
void lock(std::mutex &mutex);

void unlock(void);
void unlock(size_t &core_idx);
void unlock(std::mutex &mutex);

// Two helper functions
template <typename... Ts>
inline void add_trace(const char* name, Ts... args) {
  if (!is_enabled()) return;
  auto tc = find_traceclass(name);
  if (tc != nullptr)
    tc->add_trace(args...);
};

template <typename... Ts>
inline void add_info(const char* name, Ts... args) {
  if (!is_enabled()) return;

  find_traceclass(name)->add_info(args...);
};

string add_subgraph_raw(const xir::Subgraph* subg);
inline void add_subgraph(const xir::Subgraph* subg) {
  if (!is_enabled()) return;
  static auto tc = new_traceclass("subgraph_info", {});
  auto subgraph_info_file = add_subgraph_raw(subg);
  tc->add_info("info_file", subgraph_info_file);
};

}  // namespace vitis::ai::trace
