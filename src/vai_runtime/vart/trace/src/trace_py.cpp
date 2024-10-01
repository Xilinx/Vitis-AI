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

#include <vitis/ai/trace.hpp>
#include "str.hpp"

namespace vitis::ai::trace {

static traceClass* py_traceclass =
    new_traceclass("py", {"event_state", "py_func_name"});

extern "C" {
void tracepoint_py_func(bool start, const char* func_name) {
  if (start)
    add_trace("py", 1, std::string(func_name));
  else
    add_trace("py", 0, std::string(func_name));
}
}
}  // namespace vitis::ai::trace
