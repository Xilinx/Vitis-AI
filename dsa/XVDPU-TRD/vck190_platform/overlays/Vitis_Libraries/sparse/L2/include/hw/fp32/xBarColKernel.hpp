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
#ifndef XF_SPARSE_XBARCOLKERNEL_HPP
#define XF_SPARSE_XBARCOLKERNEL_HPP
/**
 * @file xBarColKernel.hpp
 * @brief xBarColKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscKernel.hpp"

/**
 * @brief xBarColKernel is used to select input column vector entries according to the input column pointers
 * @param in0 input axis stream of parallelly processed column vector entries
 * @param in1 input axis stream of parallelly processed column pointer entries
 * @param out output axis stream of parallelly column vector entries for the NNZs
 */
extern "C" void xBarColKernel(hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                              hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                              hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out
#if DEBUG_dumpData
                              ,
                              unsigned int p_cuId
#endif
                              );

#endif
