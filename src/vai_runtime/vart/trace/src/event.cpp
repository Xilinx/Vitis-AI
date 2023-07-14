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
#include <vitis/ai/event.hpp>
#include <vitis/ai/fmt.hpp>

#include "pid.h"
#include "time.hpp"
#include "util.hpp"
#if _WIN32
#  include <windows.h>
#else
#  include <sys/syscall.h>
#  include <sys/sysinfo.h>
#  include <sys/types.h>

#  define gettid() syscall(SYS_gettid)
#  define getpid() syscall(SYS_getpid)
#endif
namespace vitis::ai::trace {
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;

// #pragma pack(1)
traceEventBase::traceEventBase(size_t payload_size) {
#if _WIN32
  pid = GetCurrentProcessId();
#else
  pid = gettid();
#endif

#if _WIN32
  cpu_id = 0;  // TODO
#else
  cpu_id = sched_getcpu();
#endif
  ts = get_xrt_ts();
  size_ = (uint16_t)(sizeof(traceEventBase) + payload_size);
};

trace_entry_t traceEventBase::get() {
  trace_entry_t ret;
  ret.insert(make_pair("pid", to_string(pid)));
  ret.insert(make_pair("cpu_id", to_string(cpu_id)));
  ret.insert(make_pair("ts", to_string(ts)));

  return ret;
};

}  // namespace vitis::ai::trace
