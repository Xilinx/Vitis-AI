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

#ifndef XF_HPC_CG_KERNEL_UPDATEPK_HPP
#define XF_HPC_CG_KERNEL_UPDATEPK_HPP
#include "xf_blas.hpp"

typedef xf::blas::WideType<CG_dataType, CG_vecParEntries> CG_wideType;
typedef CG_wideType::t_TypeInt CG_interface;

/**
 * @brief krnl_update_pk kernel function to update vector pk
 *
 * @param p_rk the memory address to vector rk
 * @param p_pk_in the input memory address to vector pk
 * @param p_pk_out the output memory address to vector pk
 * @param p_tokenIn input stream carries the token for execution
 * @param p_tokenOut output stream carries the token for execution
 *
 */

extern "C" void krnl_update_pk(CG_interface* p_rk,
                               CG_interface* p_pk_in,
                               CG_interface* p_pk_out,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                               hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut);
#endif
