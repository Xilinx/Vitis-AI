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

/**
 * @file loadCol.hpp
 * @brief loadCol definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_LOADCOLKERNEL_HPP
#define XF_SPARSE_LOADCOLKERNEL_HPP

#include "cscKernel.hpp"

/**
 * @brief loadColKernel is used to read the input column vector and pointers out of the device memory
 * @param p_colValPtr device memory pointer for reading the input column vector
 * @param p_nnzColPtr device memory pointer for reading the column pointers of NNZ entries
 * @param out0 the output axis stream of the column vector entries
 * @param out1 the output axis stream of the column pointer entries
 */
extern "C" void loadColKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtr,
                              ap_uint<SPARSE_ddrMemBits>* p_nnzColPtr,
                              hls::stream<ap_uint<SPARSE_ddrMemBits> >& out0,
                              hls::stream<ap_uint<SPARSE_ddrMemBits> >& out1);
#endif
