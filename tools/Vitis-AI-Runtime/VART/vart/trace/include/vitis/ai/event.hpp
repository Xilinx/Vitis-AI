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
#include <unistd.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <sys/types.h>

enum TraceEventTimeType { VAI_TS_BOOT, VAI_TS_TSC, VAI_TS_XRT_NS };

namespace vitis::ai::trace {
using namespace std;

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
  pid_t pid;
  uint8_t cpu_id;
  // struct traceTimestamp ts;
  double ts;
};
#pragma pack(pop)
}  // namespace vitis::ai::trace
