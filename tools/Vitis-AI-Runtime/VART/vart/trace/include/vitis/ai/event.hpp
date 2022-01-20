/*
 * Copyright 2021 Xilinx Inc.
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
#include <sstream>
#include <string>
#include <utility>
#if _WIN32
using MY_DWORD = uint32_t;
// MSVC NOTE: it is dangerous to include windows.h in public header files. it
// introduces too many MACROS, like, max,min, CONST etc, which leads to many
// strange compilation errors. so we should avoid include windows.h in public
// header files.
// 
// #include <windows.h>
#else
#include <unistd.h>  // for getpid
#endif
enum TraceEventTimeType { VAI_TS_BOOT, VAI_TS_TSC, VAI_TS_XRT_NS };

namespace vitis::ai::trace {

// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::map;
using std::string;
using trace_entry_t = map<string, string>;

struct traceTimestamp {
  TraceEventTimeType type;
  union {
    double ts;
    uint64_t tsc;
  };
};

#pragma pack(push, 1)
class traceEventBase {
 public:
  traceEventBase(size_t payload_size = 0);
  virtual ~traceEventBase(){};
  virtual trace_entry_t get(void);
  inline size_t get_size(void) { return size_; };

 public:
  uint16_t size_;
#if _WIN32
  MY_DWORD pid;
#else
  pid_t pid;
#endif
  uint8_t cpu_id;
  // struct traceTimestamp ts;
  double ts;
};
#pragma pack(pop)
}  // namespace vitis::ai::trace
