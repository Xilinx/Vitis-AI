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

#ifndef XF_HPC_CG_CGTIMER_HPP
#define XF_HPC_CG_CGTIMER_HPP

#include "hls_stream.h"
#include "signal.hpp"
#include "timer.hpp"
using namespace std;

namespace xf {
namespace hpc {
namespace cg {

void stamp_signal(Clock_t clock, hls::stream<Signal_t>& p_signal, hls::stream<Clock_t>& p_clock) {
#pragma HLS INLINE
    Signal_t signal = IDLE;
    p_signal.read_nb(signal);
    if (signal == STAMP) {
        p_clock.write(clock);
        signal = IDLE;
    }
}

void cgTimer(hls::stream<Signal_t>& p_signal,
             hls::stream<Clock_t>& p_clock,
             hls::stream<Signal_t>& p_sigGemv,
             hls::stream<Clock_t>& p_clkGemv,
             hls::stream<Signal_t>& p_sigRk,
             hls::stream<Clock_t>& p_clkRk,
             hls::stream<Signal_t>& p_sigPk,
             hls::stream<Clock_t>& p_clkPk) {
    Signal_t signal = IDLE;
    while (signal != START) {
#pragma HLS PIPELINE
        signal = p_signal.read();
    }
    Clock_t clock = 0;
    while (signal != STOP) {
#pragma HLS PIPELINE
        clock++;

        stamp_signal(clock, p_sigGemv, p_clkGemv);
        stamp_signal(clock, p_sigRk, p_clkRk);
        stamp_signal(clock, p_sigPk, p_clkPk);

        p_signal.read_nb(signal);
        if (signal == STAMP) {
            p_clock.write(clock);
            signal = IDLE;
        }
    }
}
}
}
}
#endif
