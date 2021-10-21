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

#include "interface.hpp"
#include "krnl_timer.hpp"
using namespace std;

extern "C" void krnl_timer(hls::stream<xf::hpc::Signal_t>& p_signal, hls::stream<xf::hpc::Clock_t>& p_clock) {
    AXIS(p_signal)
    AXIS(p_clock)
    AP_CTRL_NONE(return )
#pragma HLS DATAFLOW
    xf::hpc::timer(p_signal, p_clock);
}
