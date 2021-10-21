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

#ifndef XF_HPC_TIMER_HPP
#define XF_HPC_TIMER_HPP

#include "hls_stream.h"
#include "signal.hpp"
#include "ap_utils.h"
using namespace std;

namespace xf {
namespace hpc {

template <unsigned int t_NumStamps>
void timer(hls::stream<Clock_t>& p_clock) {
    Clock_t l_clock = 0;
    unsigned int l_count = 0;
    while (l_count < t_NumStamps) {
#pragma HLS PIPELINE
        if (p_clock.write_nb(l_clock)) {
            l_count++;
        }
        l_clock++;
    }
}

void timer(hls::stream<Signal_t>& p_signal, hls::stream<Clock_t>& p_clock) {
    Signal_t signal = IDLE;
    while (signal != START) {
#pragma HLS PIPELINE
        p_signal.read_nb(signal);
    }

    Clock_t clock = 0;
    while (signal != STOP) {
#pragma HLS PIPELINE
        clock++;
        p_signal.read_nb(signal);
        if (signal == STAMP) {
            p_clock.write(clock);
            signal = IDLE;
        }
    }
}
}
}
#endif
