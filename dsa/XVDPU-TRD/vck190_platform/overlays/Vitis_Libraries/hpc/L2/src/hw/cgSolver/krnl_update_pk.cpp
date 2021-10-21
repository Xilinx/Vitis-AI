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
#include "update_pk.hpp"
#include "krnl_update_pk.hpp"

extern "C" void krnl_update_pk(CG_interface* p_pk_in,
                               CG_interface* p_pk_out,
                               CG_interface* p_zk,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut) {
    POINTER(p_pk_in, gmem_pk_in)
    POINTER(p_pk_out, gmem_pk_out)
    POINTER(p_zk, gmem_zk)
    AXIS(p_tokenIn)
    AXIS(p_tokenOut)
    SCALAR(return )

#pragma HLS DATAFLOW
    xf::hpc::cg::update_pk<CG_dataType, CG_vecParEntries, CG_tkStrWidth>(p_zk, p_pk_in, p_pk_out, p_tokenIn,
                                                                         p_tokenOut);
}
