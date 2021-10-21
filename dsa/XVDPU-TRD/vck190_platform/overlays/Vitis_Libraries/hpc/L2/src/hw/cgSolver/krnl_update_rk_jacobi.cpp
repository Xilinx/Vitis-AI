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
#include "update_rk_jacobi.hpp"
#include "krnl_update_rk.hpp"

extern "C" void krnl_update_rk_jacobi(CG_interface* p_rk_in,
                                      CG_interface* p_rk_out,
                                      CG_interface* p_zk,
                                      CG_interface* p_jacobi,
                                      CG_interface* p_Apk,
                                      hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                                      hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut) {
    POINTER(p_rk_in, gmem_rk_in)
    POINTER(p_rk_out, gmem_rk_out)
    POINTER(p_Apk, gmem_Apk)
    POINTER(p_jacobi, gmem_jacobi)
    POINTER(p_zk, gmem_zk)
    AXIS(p_tokenIn)
    AXIS(p_tokenOut)
    SCALAR(return )

#pragma HLS DATAFLOW
    xf::hpc::cg::update_rk<CG_dataType, CG_vecParEntries, CG_tkStrWidth>(p_rk_in, p_rk_out, p_zk, p_jacobi, p_Apk,
                                                                         p_tokenIn, p_tokenOut);
}
