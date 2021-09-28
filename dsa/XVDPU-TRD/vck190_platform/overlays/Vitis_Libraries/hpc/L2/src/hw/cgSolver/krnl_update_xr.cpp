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
#include "update_xr.hpp"
#include "krnl_update_xr.hpp"

extern "C" void krnl_update_xr(CG_interface* p_xk_in,
                               CG_interface* p_xk_out,
                               CG_interface* p_rk_in,
                               CG_interface* p_rk_out,
                               CG_interface* p_pk,
                               CG_interface* p_Apk,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut) {
    POINTER(p_xk_in, gmem_xk_in)
    POINTER(p_xk_out, gmem_xk_out)
    POINTER(p_rk_in, gmem_rk_in)
    POINTER(p_rk_out, gmem_rk_out)
    POINTER(p_pk, gmem_pk)
    POINTER(p_Apk, gmem_Apk)
    AXIS(p_tokenIn)
    AXIS(p_tokenOut)
    SCALAR(return )

    xf::hpc::cg::update_xr<CG_dataType, CG_parEntries, CG_tkStrWidth>(p_xk_in, p_xk_out, p_rk_in, p_rk_out, p_pk, p_Apk,
                                                                      p_tokenIn, p_tokenOut);
}
