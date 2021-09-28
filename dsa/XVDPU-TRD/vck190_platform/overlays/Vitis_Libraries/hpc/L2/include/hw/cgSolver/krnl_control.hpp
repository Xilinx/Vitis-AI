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

#ifndef XF_HPC_CG_KERNEL_CONTROL_HPP
#define XF_HPC_CG_KERNEL_CONTROL_HPP
#include "xf_blas.hpp"

typedef xf::blas::WideType<CG_dataType, CG_parEntries> CG_wideType;
typedef CG_wideType::t_TypeInt CG_interface;

/**
 * @brief krnl_control kernel function to load instructions and control the cg
 * solver
 *
 * @param p_instr the memory address to instructions
 *
 */
extern "C" void krnl_control(CG_interface* p_instr,
                             hls::stream<uint8_t>& p_signal,
                             hls::stream<uint64_t>& p_clock,
                             hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                             hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut);

#endif
