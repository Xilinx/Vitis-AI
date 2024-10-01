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

#include <chrono>
#include <utility>

#pragma once

#ifdef ENABLE_XRT_TIMESTAMP
namespace xrt_core {
unsigned long time_ns();
}
#endif

namespace vitis::ai::trace {

enum TraceEventTimeType { VAI_TS_BOOT, VAI_TS_TSC, VAI_TS_XRT_NS };

double get_ts(void);
double get_xrt_ts(void);

}  // namespace vitis::ai::trace
