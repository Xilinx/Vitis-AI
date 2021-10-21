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
#include "krnl_control.hpp"
#include "control.hpp"

extern "C" void krnl_control(CG_interface* p_instr,
                             hls::stream<uint8_t>& p_signal,
                             hls::stream<uint64_t>& p_clock,
                             hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                             hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut) {
    POINTER(p_instr, gmem)
    AXIS(p_signal)
    AXIS(p_clock)
    AXIS(p_tokenIn)
    AXIS(p_tokenOut)
    SCALAR(return )

#pragma HLS DATAFLOW
    xf::hpc::cg::control<CG_dataType, CG_parEntries, CG_instrBytes, CG_numTasks, CG_tkStrWidth>(
        p_instr, p_signal, p_clock, p_tokenIn, p_tokenOut);
}
