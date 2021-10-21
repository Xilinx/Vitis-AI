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

#ifndef XF_HPC_CG_KERNEL_GEMV_HPP
#define XF_HPC_CG_KERNEL_GEMV_HPP
#include "xf_blas.hpp"

typedef xf::blas::WideType<CG_dataType, CG_parEntries> CG_wideType;
typedef CG_wideType::t_TypeInt CG_interface;

typedef xf::blas::WideType<CG_dataType, CG_vecParEntries> CG_vecType;
typedef CG_vecType::t_TypeInt CG_vecInterface;

/**
 * @brief krnl_gemv kernel function to compute A * p
 *
 * @param p_A the memory address to vector A
 * @param p_pk the input memory address to vector pk
 * @param p_Apk the output memory address to vector Apk
 * @param p_tokenIn input stream carries the token for execution
 * @param p_tokenOut output stream carries the token for execution
 *
 */

extern "C" void krnl_gemv(CG_interface* p_A0,
#if CG_numChannels > 1
                          CG_interface* p_A1,
#endif
#if CG_numChannels > 2
                          CG_interface* p_A2,
                          CG_interface* p_A3,
#endif
#if CG_numChannels > 4
                          CG_interface* p_A4,
                          CG_interface* p_A5,
                          CG_interface* p_A6,
                          CG_interface* p_A7,
#endif
#if CG_numChannels > 8
                          CG_interface* p_A8,
                          CG_interface* p_A9,
                          CG_interface* p_Aa,
                          CG_interface* p_Ab,
                          CG_interface* p_Ac,
                          CG_interface* p_Ad,
                          CG_interface* p_Ae,
                          CG_interface* p_Af,
#endif
                          CG_interface* p_pk,
                          CG_vecInterface* p_pkc,
                          CG_vecInterface* p_Apk,
                          hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenInA,
                          hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut);

#endif
