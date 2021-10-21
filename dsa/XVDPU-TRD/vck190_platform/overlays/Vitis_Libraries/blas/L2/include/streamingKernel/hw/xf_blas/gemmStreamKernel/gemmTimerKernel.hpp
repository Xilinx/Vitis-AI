/**********
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
 * **********/

#ifndef XF_BLAS_GEMM_TIMER_KERNELS_HPP
#define XF_BLAS_GEMM_TIMER_KERNELS_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "blasInstr.hpp"
#include "gemmMatMoverL2.hpp"
namespace xf {
namespace blas {

void gemmLdStTimer(hls::stream<ap_uint<16> >& p_inStr, hls::stream<ap_uint<32> >& p_outStr) {
    ap_uint<32> l_cycles = 0;
    ap_uint<16> l_opCode = p_inStr.read();
    while (l_opCode != OpCodeType::OpControl) {
#pragma HLS PIPELINE
        (void)p_inStr.read_nb(l_opCode);
        l_cycles++;
    }
    p_outStr.write(l_cycles);
}
}
}
#endif
