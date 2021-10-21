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

void loadColKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtr,
                   ap_uint<SPARSE_ddrMemBits>* p_nnzColPtr,
                   hls::stream<ap_uint<SPARSE_ddrMemBits> >& out0,
                   hls::stream<ap_uint<SPARSE_ddrMemBits> >& out1) {
    xf::sparse::loadCol<SPARSE_maxParamDdrBlocks, SPARSE_hbmChannels, SPARSE_ddrMemBits, SPARSE_dataBits,
                        SPARSE_parEntries, SPARSE_paramOffset>(p_colValPtr, p_nnzColPtr, out0, out1);
}

void bufTransColVecKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out1,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out2,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out3,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out4,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out5,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out6,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out7,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out8,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out9,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out10,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out11,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out12,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out13,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out14,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out15) {
    const static unsigned int t_DataBits = SPARSE_dataBits * SPARSE_parEntries;

#pragma HLS DATAFLOW
    hls::stream<ap_uint<t_DataBits> > l_datStr[SPARSE_hbmChannels];

    xf::sparse::bufTransColVec<SPARSE_maxColMemBlocks, SPARSE_hbmChannels, SPARSE_parEntries, SPARSE_ddrMemBits,
                               SPARSE_dataBits>(in0, l_datStr);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[0], out0);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[1], out1);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[2], out2);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[3], out3);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[4], out4);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[5], out5);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[6], out6);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[7], out7);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[8], out8);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[9], out9);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[10], out10);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[11], out11);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[12], out12);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[13], out13);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[14], out14);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[15], out15);
}

void bufTransNnzColKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out0,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out1,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out2,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out3,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out4,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out5,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out6,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out7,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out8,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out9,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out10,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out11,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out12,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out13,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out14,
                          hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out15) {
    const static unsigned int t_DataBits = SPARSE_dataBits * SPARSE_parEntries;

#pragma HLS DATAFLOW
    hls::stream<ap_uint<t_DataBits> > l_datStr[SPARSE_hbmChannels];
    xf::sparse::bufTransNnzCol<SPARSE_maxColMemBlocks, SPARSE_hbmChannels, SPARSE_parEntries, SPARSE_ddrMemBits,
                               SPARSE_dataBits>(in0, l_datStr);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[0], out0);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[1], out1);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[2], out2);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[3], out3);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[4], out4);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[5], out5);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[6], out6);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[7], out7);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[8], out8);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[9], out9);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[10], out10);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[11], out11);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[12], out12);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[13], out13);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[14], out14);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[15], out15);
}

void xBarColKernel(hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out
#if DEBUG_dumpData
                   ,
                   unsigned int p_cuId
#endif
                   ) {

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

void cscRowKernel(hls::stream<ap_uint<SPARSE_hbmMemBits> >& in0,
                  hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
#if DEBUG_dumpData
                  hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out,
                  unsigned int p_cuId);
#else
                  hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out) {
#endif

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

extern "C" {
void cscmvSeqKernel(ap_uint<SPARSE_ddrMemBits>* p_colValPtrLoadCol,
                    ap_uint<SPARSE_ddrMemBits>* p_nnzColPtrLoadCol,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx0,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes0,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx1,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes1,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx2,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes2,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx3,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes3,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx4,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes4,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx5,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes5,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx6,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes6,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx7,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes7,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx8,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes8,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx9,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes9,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx10,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes10,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx11,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes11,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx12,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes12,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx13,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes13,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx14,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes14,
                    ap_uint<SPARSE_hbmMemBits>* p_aNnzIdx15,
                    ap_uint<SPARSE_hbmMemBits>* p_rowRes15) {
#pragma HLS INTERFACE m_axi port = p_colValPtrLoadCol offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_nnzColPtrLoadCol offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = p_aNnzIdx0 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = p_aNnzIdx1 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = p_aNnzIdx2 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = p_aNnzIdx3 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = p_aNnzIdx4 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = p_aNnzIdx5 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = p_aNnzIdx6 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = p_aNnzIdx7 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = p_aNnzIdx8 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = p_aNnzIdx9 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = p_aNnzIdx10 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = p_aNnzIdx11 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = p_aNnzIdx12 offset = slave bundle = gmem14
#pragma HLS INTERFACE m_axi port = p_aNnzIdx13 offset = slave bundle = gmem15
#pragma HLS INTERFACE m_axi port = p_aNnzIdx14 offset = slave bundle = gmem16
#pragma HLS INTERFACE m_axi port = p_aNnzIdx15 offset = slave bundle = gmem17

#pragma HLS INTERFACE m_axi port = p_rowRes0 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = p_rowRes1 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = p_rowRes2 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = p_rowRes3 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = p_rowRes4 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = p_rowRes5 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = p_rowRes6 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = p_rowRes7 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = p_rowRes8 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = p_rowRes9 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = p_rowRes10 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = p_rowRes11 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = p_rowRes12 offset = slave bundle = gmem14
#pragma HLS INTERFACE m_axi port = p_rowRes13 offset = slave bundle = gmem15
#pragma HLS INTERFACE m_axi port = p_rowRes14 offset = slave bundle = gmem16
#pragma HLS INTERFACE m_axi port = p_rowRes15 offset = slave bundle = gmem17

#pragma HLS INTERFACE s_axilite port = p_colValPtrLoadCol bundle = control
#pragma HLS INTERFACE s_axilite port = p_nnzColPtrLoadCol bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx3 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx4 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx5 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx6 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx7 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx8 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx9 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx10 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx11 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx12 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx13 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx14 bundle = control
#pragma HLS INTERFACE s_axilite port = p_aNnzIdx15 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes3 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes4 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes5 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes6 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes7 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes8 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes9 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes10 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes11 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes12 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes13 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes14 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rowRes15 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
    static const unsigned int t_ParamsPerDdr = SPARSE_ddrMemBits / 32;
    static const unsigned int t_DataBits = SPARSE_dataBits * SPARSE_parEntries;

    hls::stream<ap_uint<SPARSE_ddrMemBits> > out0LoadCol;
    hls::stream<ap_uint<SPARSE_ddrMemBits> > out1LoadCol;
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > outBufColVec[SPARSE_hbmChannels];
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > outBufColPtr[SPARSE_hbmChannels];

    loadColKernel(p_colValPtrLoadCol, p_nnzColPtrLoadCol, out0LoadCol, out1LoadCol);
    WideType<uint32_t, t_ParamsPerDdr> l_param;

    l_param = out0LoadCol.read();
    (void)out1LoadCol.read();
    unsigned int l_totalParams = l_param[0];
    unsigned int l_iterations = l_param[1];

    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > outXbarCol[SPARSE_hbmChannels];
    for (unsigned int r = 0; r < l_totalParams * l_iterations; ++r) {
        bufTransColVecKernel(out0LoadCol, outBufColVec[0], outBufColVec[1], outBufColVec[2], outBufColVec[3],
                             outBufColVec[4], outBufColVec[5], outBufColVec[6], outBufColVec[7], outBufColVec[8],
                             outBufColVec[9], outBufColVec[10], outBufColVec[11], outBufColVec[12], outBufColVec[13],
                             outBufColVec[14], outBufColVec[15]);
        bufTransNnzColKernel(out1LoadCol, outBufColPtr[0], outBufColPtr[1], outBufColPtr[2], outBufColPtr[3],
                             outBufColPtr[4], outBufColPtr[5], outBufColPtr[6], outBufColPtr[7], outBufColPtr[8],
                             outBufColPtr[9], outBufColPtr[10], outBufColPtr[11], outBufColPtr[12], outBufColPtr[13],
                             outBufColPtr[14], outBufColPtr[15]);

        for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
#if DEBUG_dumpData
            xBarColKernel(outBufColVec[i], outBufColPtr[i], outXbarCol[i], r * SPARSE_hbmChannels + i);
#else
            xBarColKernel(outBufColVec[i], outBufColPtr[i], outXbarCol[i]);
#endif
        }
    }

    hls::stream<ap_uint<SPARSE_hbmMemBits> > rdWrHbmOut[SPARSE_hbmChannels];
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > rdWrHbmIn[SPARSE_hbmChannels];

    hls::stream<uint32_t> l_paramStr[SPARSE_hbmChannels];
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx0, l_paramStr[0],
                                                                                         rdWrHbmOut[0]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx1, l_paramStr[1],
                                                                                         rdWrHbmOut[1]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx2, l_paramStr[2],
                                                                                         rdWrHbmOut[2]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx3, l_paramStr[3],
                                                                                         rdWrHbmOut[3]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx4, l_paramStr[4],
                                                                                         rdWrHbmOut[4]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx5, l_paramStr[5],
                                                                                         rdWrHbmOut[5]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx6, l_paramStr[6],
                                                                                         rdWrHbmOut[6]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx7, l_paramStr[7],
                                                                                         rdWrHbmOut[7]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx8, l_paramStr[8],
                                                                                         rdWrHbmOut[8]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx9, l_paramStr[9],
                                                                                         rdWrHbmOut[9]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx10, l_paramStr[10],
                                                                                         rdWrHbmOut[10]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx11, l_paramStr[11],
                                                                                         rdWrHbmOut[11]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx12, l_paramStr[12],
                                                                                         rdWrHbmOut[12]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx13, l_paramStr[13],
                                                                                         rdWrHbmOut[13]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx14, l_paramStr[14],
                                                                                         rdWrHbmOut[14]);
    xf::sparse::readHbm<SPARSE_maxParamHbmBlocks, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_aNnzIdx15, l_paramStr[15],
                                                                                         rdWrHbmOut[15]);

    for (unsigned int it = 0; it < l_iterations; ++it) {
        for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
#if DEBUG_dumpData
            cscRowKernel(rdWrHbmOut[i], outXbarCol[i], rdWrHbmIn[i], i);
#else
            cscRowKernel(rdWrHbmOut[i], outXbarCol[i], rdWrHbmIn[i]);
#endif
        }
    }

    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[0], rdWrHbmIn[0], p_rowRes0);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[1], rdWrHbmIn[1], p_rowRes1);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[2], rdWrHbmIn[2], p_rowRes2);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[3], rdWrHbmIn[3], p_rowRes3);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[4], rdWrHbmIn[4], p_rowRes4);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[5], rdWrHbmIn[5], p_rowRes5);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[6], rdWrHbmIn[6], p_rowRes6);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[7], rdWrHbmIn[7], p_rowRes7);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[8], rdWrHbmIn[8], p_rowRes8);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[9], rdWrHbmIn[9], p_rowRes9);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[10], rdWrHbmIn[10], p_rowRes10);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[11], rdWrHbmIn[11], p_rowRes11);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[12], rdWrHbmIn[12], p_rowRes12);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[13], rdWrHbmIn[13], p_rowRes13);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[14], rdWrHbmIn[14], p_rowRes14);
    xf::sparse::writeHbm<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_hbmMemBits, SPARSE_dataBits>(
        l_paramStr[15], rdWrHbmIn[15], p_rowRes15);

#ifndef __SYNTHESIS__
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        if (!rdWrHbmIn[i].empty()) {
            std::cout << "ERROR: in cscmvSeqKernel, rdWrHbmIn[" << i << "] is not empty" << std::endl;
        }
    }
#endif
}
}
