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
#ifndef XF_HPC_CG_KERNEL_DEF_HPP
#define XF_HPC_CG_KERNEL_DEF_HPP
typedef xf::blas::WideType<CG_dataType, CG_parEntries> CG_wideType;
typedef CG_wideType::t_TypeInt CG_interface;
typedef xf::blas::WideType<CG_dataType, CG_vecParEntries> CG_vecType;
typedef CG_vecType::t_TypeInt CG_vecInterface;
typedef hls::stream<uint32_t> CG_paramStrType;
typedef hls::stream<ap_uint<32 * CG_numChannels> > CG_wideParamStrType;
typedef hls::stream<CG_interface> CG_wideStrType;
typedef hls::stream<ap_uint<SPARSE_dataBits> > CG_datStrType;
typedef hls::stream<ap_uint<CG_tkStrWidth> > CG_tkStrType;
typedef hls::stream<ap_uint<SPARSE_indexBits> > CG_idxStrType;
#endif
