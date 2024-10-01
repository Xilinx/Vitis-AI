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

#include <mutex>
#include <vitis/ai/payload.hpp>
#include <vitis/ai/ringbuf.hpp>
#include <vitis/ai/traceclass.hpp>

namespace vitis::ai::trace {

extern bool is_enabled(void);

std::mutex table_lock;
vector<traceClass*> traceclass_table;

static traceClass time_sync("trace_timesync", {});
static traceClass dpu_controller("dpu-controller", {"event_state", "device_core_idx", "hwconuter"});
static traceClass dpu_runner("dpu-runner", {"subgraph", "batch", "workload", "depth"});
static traceClass cpu_task("cpu-task", {"subgraph", "depth", "event_state"});

traceClass::traceClass(const char* name_, vector<string> items) {
  classname = std::string(name_);
  column_names = items;
  column_num = items.size();

  table_lock.lock();
  traceclass_table.push_back(this);
  table_lock.unlock();
};

traceClass* new_traceclass(const char* name_, vector<string> items) {
  if (!is_enabled()) {
    return nullptr;
  }

  CHECK (find_traceclass(name_) == nullptr);

  auto tc = new traceClass(name_, items);

  return tc;
};

traceClass* find_traceclass(const char* name) {
  if (!is_enabled()) {
    return nullptr;
  }

  // Only lock table for write
  for (const auto t : traceclass_table) {
    if (strcmp(t->classname.c_str(), name) == 0) return t;
  }

  return nullptr;
};

}  // namespace vitis::ai::trace
