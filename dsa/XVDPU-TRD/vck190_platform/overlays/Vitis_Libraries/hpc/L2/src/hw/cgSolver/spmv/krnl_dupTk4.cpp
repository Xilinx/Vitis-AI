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

extern "C" void krnl_dupTk4(hls::stream<ap_uint<CG_tkStrWidth> >& p_tkInStr,
                            hls::stream<ap_uint<CG_tkStrWidth> >& p_tkOutStr0,
                            hls::stream<ap_uint<CG_tkStrWidth> >& p_tkOutStr1,
                            hls::stream<ap_uint<CG_tkStrWidth> >& p_tkOutStr2,
                            hls::stream<ap_uint<CG_tkStrWidth> >& p_tkOutStr3) {
    AXIS(p_tkInStr)
    AXIS(p_tkOutStr0)
    AXIS(p_tkOutStr1)
    AXIS(p_tkOutStr2)
    AXIS(p_tkOutStr3)
    AP_CTRL_NONE(return )

#pragma HLS STREAM variable = p_tkInStr depth = 64
#pragma HLS PIPELINE
    ap_uint<CG_tkStrWidth> l_val = p_tkInStr.read();
    p_tkOutStr0.write(l_val);
    p_tkOutStr1.write(l_val);
    p_tkOutStr2.write(l_val);
    p_tkOutStr3.write(l_val);
}
