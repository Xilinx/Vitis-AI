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
 * @file cscmvSeqKernel.cpp
 * @brief loadCol definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscKernel.hpp"

extern "C" {
void loadColKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtr,
                   ap_uint<SPARSE_ddrMemBits>* p_nnzColPtr,
                   hls::stream<ap_uint<SPARSE_ddrMemBits> >& out0,
                   hls::stream<ap_uint<SPARSE_ddrMemBits> >& out1) {
#pragma HLS INLINE off
    xf::sparse::loadCol<SPARSE_maxParamDdrBlocks, SPARSE_hbmChannels, SPARSE_ddrMemBits, SPARSE_dataBits,
                        SPARSE_parEntries, SPARSE_paramOffset>(p_colValPtr, p_nnzColPtr, out0, out1);
}
void bufTransColVecKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > out[SPARSE_hbmChannels]) {
    xf::sparse::bufTransColVec<SPARSE_maxColMemBlocks, SPARSE_hbmChannels, SPARSE_parEntries, SPARSE_ddrMemBits,
                               SPARSE_dataBits>(in0, out);
}
void bufTransNnzColKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > out[SPARSE_hbmChannels]) {
    xf::sparse::bufTransNnzCol<SPARSE_maxColMemBlocks, SPARSE_hbmChannels, SPARSE_parEntries, SPARSE_ddrMemBits,
                               SPARSE_dataBits>(in0, out);
}
void xBarColKernel(hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out) {
    const static unsigned int t_numParams = SPARSE_dataBits * SPARSE_parEntries / 32;

    ap_uint<SPARSE_dataBits* SPARSE_parEntries> l_param0 = in0.read();
    ap_uint<SPARSE_dataBits* SPARSE_parEntries> l_param1 = in1.read();
    WideType<unsigned int, t_numParams> l_param0Val(l_param0);
    WideType<unsigned int, t_numParams> l_param1Val(l_param1);

    unsigned int l_colPtrBlocks = l_param0Val[1];
    unsigned int l_nnzBlocks = l_param1Val[1];
    xf::sparse::xBarColUnit<SPARSE_logParEntries, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits>(
        l_colPtrBlocks, l_nnzBlocks, in0, in1, out);
}

void cscRowKernel(hls::stream<ap_uint<SPARSE_hbmMemBits> >& in0,
                  hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                  hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out) {
    const static unsigned int t_numParams = SPARSE_hbmMemBits / 32;

    ap_uint<SPARSE_hbmMemBits> l_param = in0.read();
    WideType<uint32_t, t_numParams> l_paramVal(l_param);

    unsigned int l_totalParams = l_paramVal[0];

    for (unsigned int i = 0; i < l_totalParams; ++i) {
        l_paramVal = in0.read();

        unsigned int l_nnzBlocks = l_paramVal[0];
        unsigned int l_rowBlocks = l_paramVal[1];
        xf::sparse::cscRowUnit<SPARSE_maxRowBlocks, SPARSE_logParEntries, SPARSE_logParGroups, SPARSE_dataType,
                               SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits, SPARSE_hbmMemBits>(
            in0, in1, l_nnzBlocks, l_rowBlocks, out);
    }
}

void cscmvSeqKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtrLoadCol,
                    ap_uint<SPARSE_ddrMemBits>* p_nnzColPtrLoadCol,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx0,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes0) {
#pragma HLS INTERFACE m_axi port = p_colValPtrLoadCol offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_nnzColPtrLoadCol offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = p_aNnzIdx0 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = p_rowRes0 offset = slave bundle = gmem2

#pragma HLS INTERFACE s_axilite port = p_colValPtrLoadCol bundle = control
#pragma HLS INTERFACE s_axilite port = p_nnzColPtrLoadCol bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes0 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

    static const unsigned int t_ParamsPerDdr = SPARSE_ddrMemBits / 32;
    static const unsigned int t_DataBits = SPARSE_dataBits * SPARSE_parEntries;

    hls::stream<ap_uint<SPARSE_ddrMemBits> > out0LoadCol;
    hls::stream<ap_uint<SPARSE_ddrMemBits> > out1LoadCol;
    hls::stream<ap_uint<t_DataBits> > outBufColVec[SPARSE_hbmChannels];
    hls::stream<ap_uint<t_DataBits> > outBufColPtr[SPARSE_hbmChannels];
    hls::stream<ap_uint<t_DataBits> > outXbarCol[SPARSE_hbmChannels];
    hls::stream<ap_uint<SPARSE_hbmMemBits> > rdWrHbmOut[SPARSE_hbmChannels];
    hls::stream<ap_uint<t_DataBits> > rdWrHbmIn[SPARSE_hbmChannels];
    hls::stream<uint32_t> l_paramStr[SPARSE_hbmChannels];

#pragma HLS DATAFLOW
    loadColKernel(p_colValPtrLoadCol, p_nnzColPtrLoadCol, out0LoadCol, out1LoadCol);

    bufTransColVecKernel(out0LoadCol, outBufColVec);
    bufTransNnzColKernel(out1LoadCol, outBufColPtr);

    xBarColKernel(outBufColVec[0], outBufColPtr[0], outXbarCol[0]);

    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx0, l_paramStr[0],
                                                                                         rdWrHbmOut[0]);

    cscRowKernel(rdWrHbmOut[0], outXbarCol[0], rdWrHbmIn[0]);

    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[0], rdWrHbmIn[0], p_rowRes0);
}
}
