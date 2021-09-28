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
#ifndef XF_SPARSE_CSCROWPKT_HPP
#define XF_SPARSE_CSCROWPKT_HPP
/**
 * @file cscRowPktKernel.hpp
 * @brief cscRowPktKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscKernel.hpp"

/**
 * @brief cscRowKernel is used to accumulate the multiplication results for the same row
 * @param in0 the input axis stream of the NNZs' values and row indices
 * @param in1 the input axis stream of column vector entries for the NNZs
 * @param out the output axis stream of result row vector entries
 */
extern "C" void cscRowKernel(hls::stream<ap_uint<SPARSE_hbmMemBits> >& in0,
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
#if DEBUG_dumpData
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out,
                             unsigned int p_cuId);
#else
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out);
#endif
#endif
