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
 * @file rowAccKernel.cpp
 * @brief rowAccKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */
#include "rowAccKernel.hpp"

void fwdRbParam(WideParamStrTyp& p_inParamStr, ParamStrTyp p_outParamStr[SPARSE_hbmChannels]) {
    uint32_t l_numRbs = p_inParamStr.read();
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
#pragma HLS UNROLL
        p_outParamStr[i].write(l_numRbs);
    }

    for (unsigned int r = 0; r < l_numRbs; ++r) {
        WideType<uint32_t, SPARSE_hbmChannels> l_val;
#pragma HLS PARTITION variable = l_val complete dim = 1
        for (unsigned int j = 0; j < 4; ++j) {
#pragma HLS PIPELINE
            l_val = p_inParamStr.read();
            for (unsigned int k = 0; k < SPARSE_hbmChannels; ++k) {
                p_outParamStr[k].write(l_val[k]);
            }
        }
    }
}

extern "C" void rowAccKernel(WideParamStrTyp& p_paramStr,

                             DatStrTyp& p_inDatStr0,
                             DatStrTyp& p_inDatStr1,
                             DatStrTyp& p_inDatStr2,
                             DatStrTyp& p_inDatStr3,
                             DatStrTyp& p_inDatStr4,
                             DatStrTyp& p_inDatStr5,
                             DatStrTyp& p_inDatStr6,
                             DatStrTyp& p_inDatStr7,
                             DatStrTyp& p_inDatStr8,
                             DatStrTyp& p_inDatStr9,
                             DatStrTyp& p_inDatStr10,
                             DatStrTyp& p_inDatStr11,
                             DatStrTyp& p_inDatStr12,
                             DatStrTyp& p_inDatStr13,
                             DatStrTyp& p_inDatStr14,
                             DatStrTyp& p_inDatStr15,

                             IdxStrTyp& p_idxStr0,
                             IdxStrTyp& p_idxStr1,
                             IdxStrTyp& p_idxStr2,
                             IdxStrTyp& p_idxStr3,
                             IdxStrTyp& p_idxStr4,
                             IdxStrTyp& p_idxStr5,
                             IdxStrTyp& p_idxStr6,
                             IdxStrTyp& p_idxStr7,
                             IdxStrTyp& p_idxStr8,
                             IdxStrTyp& p_idxStr9,
                             IdxStrTyp& p_idxStr10,
                             IdxStrTyp& p_idxStr11,
                             IdxStrTyp& p_idxStr12,
                             IdxStrTyp& p_idxStr13,
                             IdxStrTyp& p_idxStr14,
                             IdxStrTyp& p_idxStr15,

                             DatStrTyp& p_outDatStr) {
    AXIS(p_paramStr)

    AXIS(p_inDatStr0)
    AXIS(p_inDatStr1)
    AXIS(p_inDatStr2)
    AXIS(p_inDatStr3)
    AXIS(p_inDatStr4)
    AXIS(p_inDatStr5)
    AXIS(p_inDatStr6)
    AXIS(p_inDatStr7)
    AXIS(p_inDatStr8)
    AXIS(p_inDatStr9)
    AXIS(p_inDatStr10)
    AXIS(p_inDatStr11)
    AXIS(p_inDatStr12)
    AXIS(p_inDatStr13)
    AXIS(p_inDatStr14)
    AXIS(p_inDatStr15)

    AXIS(p_idxStr0)
    AXIS(p_idxStr1)
    AXIS(p_idxStr2)
    AXIS(p_idxStr3)
    AXIS(p_idxStr4)
    AXIS(p_idxStr5)
    AXIS(p_idxStr6)
    AXIS(p_idxStr7)
    AXIS(p_idxStr8)
    AXIS(p_idxStr9)
    AXIS(p_idxStr10)
    AXIS(p_idxStr11)
    AXIS(p_idxStr12)
    AXIS(p_idxStr13)
    AXIS(p_idxStr14)
    AXIS(p_idxStr15)

    AXIS(p_outDatStr)
    AP_CTRL_NONE(return )

    DatStrTyp l_datStr[SPARSE_hbmChannels];
    ParamStrTyp l_paramStr[SPARSE_hbmChannels];
    static const unsigned int t_Xdepth = SPARSE_maxRows * 2;
#pragma HLS STREAM variable = l_datStr depth = t_Xdepth
#pragma HLS RESOURCE variable = l_datStr core = fifo_uram
#pragma HLS STREAM variable = l_paramStr depth = 32
#pragma HLS DATAFLOW

    fwdRbParam(p_paramStr, l_paramStr);

    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[0], p_inDatStr0, p_idxStr0, l_datStr[0]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[1], p_inDatStr1, p_idxStr1, l_datStr[1]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[2], p_inDatStr2, p_idxStr2, l_datStr[2]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[3], p_inDatStr3, p_idxStr3, l_datStr[3]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[4], p_inDatStr4, p_idxStr4, l_datStr[4]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[5], p_inDatStr5, p_idxStr5, l_datStr[5]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[6], p_inDatStr6, p_idxStr6, l_datStr[6]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[7], p_inDatStr7, p_idxStr7, l_datStr[7]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[8], p_inDatStr8, p_idxStr8, l_datStr[8]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[9], p_inDatStr9, p_idxStr9, l_datStr[9]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[10], p_inDatStr10, p_idxStr10, l_datStr[10]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[11], p_inDatStr11, p_idxStr11, l_datStr[11]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[12], p_inDatStr12, p_idxStr12, l_datStr[12]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[13], p_inDatStr13, p_idxStr13, l_datStr[13]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[14], p_inDatStr14, p_idxStr14, l_datStr[14]);
    xf::sparse::rowAcc<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_parEntries, SPARSE_dataBits,
                       SPARSE_indexBits, SPARSE_accLatency>(l_paramStr[15], p_inDatStr15, p_idxStr15, l_datStr[15]);
    xf::sparse::accY<SPARSE_dataType, SPARSE_hbmChannels, SPARSE_dataBits>(l_datStr, p_outDatStr);
}
