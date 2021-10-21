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
#include "blasKernels.hpp"

extern "C" void gemmTimerKernel(hls::stream<ap_uint<16> >& l_opCodeStr, hls::stream<ap_uint<32> >& l_resStr) {
#pragma HLS INTERFACE axis port = l_opCodeStr bundle = l_opCodeStr
#pragma HLS INTERFACE axis port = l_resStr bundle = l_resStr
#pragma HLS INTERFACE ap_ctrl_none port = return
    xf::blas::gemmLdStTimer(l_opCodeStr, l_resStr);
}
