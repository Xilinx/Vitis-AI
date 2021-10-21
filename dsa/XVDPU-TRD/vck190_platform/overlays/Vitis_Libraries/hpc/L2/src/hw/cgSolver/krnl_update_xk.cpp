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
#include "update_xk.hpp"
#include "krnl_update_xk.hpp"

extern "C" void krnl_update_xk(CG_interface* p_xk_in,
                               CG_interface* p_xk_out,
                               CG_interface* p_pk,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn) {
    POINTER(p_xk_in, gmem_xk_in)
    POINTER(p_xk_out, gmem_xk_out)
    POINTER(p_pk, gmem_pk)
    AXIS(p_tokenIn)
    SCALAR(return )
#pragma HLS DATAFLOW

    xf::hpc::cg::update_xk<CG_dataType, CG_vecParEntries, CG_tkStrWidth>(p_xk_in, p_xk_out, p_pk, p_tokenIn);
}
