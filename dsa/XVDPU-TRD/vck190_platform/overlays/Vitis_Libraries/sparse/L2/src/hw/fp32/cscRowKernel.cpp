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
 * @file cscRowPktKernel.cpp
 * @brief cscRowPktKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscRowKernel.hpp"

extern "C" void cscRowKernel(hls::stream<ap_uint<SPARSE_hbmMemBits> >& in0,
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
#if DEBUG_dumpData
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out,
                             unsigned int p_cuId);
#else
                             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out) {
#endif
#pragma HLS INTERFACE axis port = in0
#pragma HLS INTERFACE axis port = in1
#pragma HLS INTERFACE axis port = out

#if DEBUG_dumpData
#pragma HLS INTERFACE s_axilite port = return bundle = control
#else
#pragma HLS INTERFACE ap_ctrl_none port = return
#endif

const static unsigned int t_numParams = SPARSE_hbmMemBits / 32;

WideType<uint32_t, t_numParams> l_paramVal = in0.read();

unsigned int l_totalParams = l_paramVal[0];

for (unsigned int i = 0; i < l_totalParams; ++i) {
    l_paramVal = in0.read();

    unsigned int l_nnzBlocks = l_paramVal[0];
    unsigned int l_rowBlocks = l_paramVal[1];

#if DEBUG_dumpData
    xf::sparse::cscRowUnit<SPARSE_maxRowBlocks, SPARSE_logParEntries, SPARSE_logParGroups, SPARSE_dataType,
                           SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits, SPARSE_hbmMemBits>(
        in0, in1, l_nnzBlocks, l_rowBlocks, out, i * SPARSE_hbmChannels + p_cuId);
#else
    xf::sparse::cscRowUnit<SPARSE_maxRowBlocks, SPARSE_logParEntries, SPARSE_logParGroups, SPARSE_dataType,
                           SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits, SPARSE_hbmMemBits>(
        in0, in1, l_nnzBlocks, l_rowBlocks, out);
#endif
}
}
