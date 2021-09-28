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

#ifndef HPC_CG_KERNEL_UPDATERK_HPP
#define HPC_CG_KERNEL_UPDATERK_HPP
#include "xf_blas.hpp"

typedef xf::blas::WideType<CG_dataType, CG_vecParEntries> CG_wideType;
typedef CG_wideType::t_TypeInt CG_interface;

/**
 * @brief krnl_update_xr kernel function to update the vector xk and rk
 *
 * @param p_rk_in the input memory address to vector rk
 * @param p_rk_out the output memory address to vector rk
 * @param p_Apk the memory address to vector Apk
 * @param p_tokenIn input stream carries the token for execution
 * @param p_tokenOut output stream carries the token for execution
 *
 */
extern "C" void krnl_update_rk(CG_interface* p_rk_in,
                               CG_interface* p_rk_out,
                               CG_interface* p_Apk,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut);

#endif
