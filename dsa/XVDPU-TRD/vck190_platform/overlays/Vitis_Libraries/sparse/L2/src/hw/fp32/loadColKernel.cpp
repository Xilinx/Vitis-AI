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
 * @file loadCol.cpp
 * @brief loadCol definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "loadColKernel.hpp"
extern "C" void loadColKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtr,
                              ap_uint<SPARSE_ddrMemBits>* p_nnzColPtr,
                              hls::stream<ap_uint<SPARSE_ddrMemBits> >& out0,
                              hls::stream<ap_uint<SPARSE_ddrMemBits> >& out1) {
#pragma HLS INTERFACE m_axi port = p_colValPtr offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = p_nnzColPtr offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = p_colValPtr bundle = control
#pragma HLS INTERFACE s_axilite port = p_nnzColPtr bundle = control
#pragma HLS INTERFACE axis port = out0
#pragma HLS INTERFACE axis port = out1
#pragma HLS INTERFACE s_axilite port = return bundle = control
    xf::sparse::loadCol<SPARSE_maxParamDdrBlocks, SPARSE_hbmChannels, SPARSE_ddrMemBits, SPARSE_dataBits,
                        SPARSE_parEntries, SPARSE_paramOffset>(p_colValPtr, p_nnzColPtr, out0, out1);
}
