/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_GRAPH_L3_COMMON_CPP_
#define _XF_GRAPH_L3_COMMON_CPP_

#include "common.hpp"

namespace xf {
namespace graph {
namespace L3 {

double showTimeData(std::string p_Task, TimePointType& t1, TimePointType& t2) {
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    std::cout << p_Task << "  " << std::fixed << std::setprecision(6) << l_timeMs << " msec" << std::endl;
    return l_timeMs;
};

} // L3
} // graph
} // xf

#endif
