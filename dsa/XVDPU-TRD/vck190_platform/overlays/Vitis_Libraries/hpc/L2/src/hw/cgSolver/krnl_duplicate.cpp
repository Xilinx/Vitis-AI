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

extern "C" void krnl_duplicate(hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenX,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenR) {
    AXIS(p_tokenIn)
    AXIS(p_tokenX)
    AXIS(p_tokenR)
    AP_CTRL_NONE(return )

#pragma HLS PIPELINE
    ap_uint<CG_tkStrWidth> l_val = p_tokenIn.read();
    p_tokenX.write(l_val);
    p_tokenR.write(l_val);
}
