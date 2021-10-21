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
 * @file xBarColKernel.cpp
 * @brief xBarColKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "xBarColKernel.hpp"

extern "C" void xBarColKernel(hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                              hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                              hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out
#if DEBUG_dumpData
                              ,
                              unsigned int p_cuId
#endif
                              ) {

#pragma HLS INTERFACE axis port = in0
#pragma HLS INTERFACE axis port = in1
#pragma HLS INTERFACE axis port = out

#if DEBUG_dumpData
#pragma HLS INTERFACE s_axilite port = return bundle = control
#else
#pragma HLS INTERFACE ap_ctrl_none port = return
#endif

    const static unsigned int t_numParams = SPARSE_dataBits * SPARSE_parEntries / 32;

    WideType<unsigned int, t_numParams> l_param0Val = in0.read();
    WideType<unsigned int, t_numParams> l_param1Val = in1.read();
    unsigned int l_colPtrBlocks = l_param0Val[1];
    unsigned int l_nnzBlocks = l_param1Val[1];

    if (l_nnzBlocks == 0) {
        return;
    }
    xf::sparse::xBarColUnit<SPARSE_logParEntries, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits>(
        l_colPtrBlocks, l_nnzBlocks, in0, in1, out
#if DEBUG_dumpData
        ,
        p_cuId
#endif
        );
}
