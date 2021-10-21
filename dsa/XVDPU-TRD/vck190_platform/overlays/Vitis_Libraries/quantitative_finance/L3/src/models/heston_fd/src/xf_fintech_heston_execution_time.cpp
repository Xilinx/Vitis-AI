/*
 * Copyright 2019 Xilinx, Inc.
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

#include "xf_fintech_heston_execution_time.hpp"

namespace xf {
namespace fintech {

void HestonFDExecutionTime::Start() {
    _Start = std::chrono::high_resolution_clock::now();
}

void HestonFDExecutionTime::Stop() {
    _Stop = std::chrono::high_resolution_clock::now();
}

std::chrono::milliseconds HestonFDExecutionTime::Duration() {
    std::chrono::duration<double> Duration = _Stop - _Start;
    std::chrono::milliseconds Result = std::chrono::duration_cast<std::chrono::milliseconds>(Duration);

    return Result;
}

} // namespace fintech
} // namespace xf
