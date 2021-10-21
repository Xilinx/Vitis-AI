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
#include "top.hpp"
#include "token.hpp"
#include "update_rk_jacobi.hpp"

void top(ap_uint<CG_memBits> p_rk_in[CG_dataSize / CG_parEntries],
         ap_uint<CG_memBits> p_rk_out[CG_dataSize / CG_parEntries],
         ap_uint<CG_memBits> p_zk[CG_dataSize / CG_parEntries],
         ap_uint<CG_memBits> p_jacobi[CG_dataSize / CG_parEntries],
         ap_uint<CG_memBits> p_Apk[CG_dataSize / CG_parEntries],
         hls::stream<ap_uint<CG_tkWidth> >& p_tokenIn,
         hls::stream<ap_uint<CG_tkWidth> >& p_tokenOut) {
    xf::hpc::cg::update_rk<CG_dataType, CG_parEntries, CG_tkWidth>(p_rk_in, p_rk_out, p_zk, p_jacobi, p_Apk, p_tokenIn,
                                                                   p_tokenOut);
}
