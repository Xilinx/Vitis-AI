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

#ifndef _XF_GRAPH_CONVERT_CSR_CSC_HPP_
#define _XF_GRAPH_CONVERT_CSR_CSC_HPP_

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include <hls_stream.h>
#include <ap_int.h>
#include "L2_utils.hpp"
#include "calc_degree.hpp"

namespace xf {
namespace graph {
namespace internal {
namespace convert_csr_csc {

template <typename DT, typename uint512, int K = 16, int W = 32>
void calcuOffset(int vertexLen, uint512* degree, uint512* offset, uint512* offsetTmp) {
    DT s = 0;
loop_calcuOffset:
    for (int i = 0; i < (vertexLen + K) / K; i++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 4
        uint512 degree_tmp = degree[i];
        uint512 offset_tmp = 0;
        for (int j = 0; j < K; j++) {
            DT tmp = degree_tmp.range(W * (j + 1) - 1, W * j);
            offset_tmp.range(W * (j + 1) - 1, W * j) = s;
            s += tmp;
        }
        offset[i] = offset_tmp;
        offsetTmp[i] = offset_tmp;
    }
}

#ifndef __SYNTHESIS__
template <typename DT,
          typename uint512,
          int rowTemplate,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,
          int K = 16,
          int W = 32>
void calcuIndex(int vertexLen,
                ap_uint<dataOneLine>** valid,
                uint512** onChipUram,
                ap_uint<addrWidth>** onChipAddr,
                hls::stream<uint512>& indexG1Strm,
                hls::stream<uint512>& offsetG1Strm,
                // DT* indexG2,
                uint512* offsetG2,
                hls::stream<DT>& addrG2Strm,
                hls::stream<DT>& indexG2Strm) {
#else
template <typename DT,
          typename uint512,
          int rowTemplate,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,
          int K = 16,
          int W = 32>
void calcuIndex(int vertexLen,
                ap_uint<dataOneLine> valid[uramRow][groupUramPart],
                uint512 onChipUram[uramRow][groupUramPart],
                ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart],
                hls::stream<uint512>& indexG1Strm,
                hls::stream<uint512>& offsetG1Strm,
                // DT* indexG2,
                uint512* offsetG2,
                hls::stream<DT>& addrG2Strm,
                hls::stream<DT>& indexG2Strm) {
#endif
    ap_uint<1024> tmpIn;
    tmpIn.range(1023, 512) = offsetG1Strm.read();
    DT begin = 0;
    DT end = 0;
    uint512 indexG1Pre;
    DT indexG1IndexPre = 0xffffffff;
    const int iteration = (vertexLen + K - 1) / K;

    //// cache init
    ap_uint<dataOneLine> validCnt = -1;
    uint512 tmp = 0;
Loop_init_uram:
    for (int j = 0; j < uramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
        for (int i = 0; i < groupUramPart; ++i) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16
#pragma HLS pipeline II = 1
            valid[j][i] = 0;
            onChipAddr[j][i] = 0;
            onChipUram[j][i] = 0;
        }
    }

    ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
    uint512 pingQue[4] = {0, 0, 0, 0};
    uint512 cntQue[4] = {0, 0, 0, 0};
    ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
    int validAddrQue[4] = {-1, -1, -1, -1};
    int addrAddrQue[4] = {-1, -1, -1, -1};
    int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

    DT k00, k01, k10, k11, k20, k21, k30;
//////////////

loop_calcuIndex0:
    for (int i = 0; i < vertexLen; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
        if ((i + 1) % K == 0)
            tmpIn.range(1023, 512) = offsetG1Strm.read();
        else
            tmpIn.range(511, 0) = tmpIn.range(1023, 512);
        end = tmpIn.range(((i % K) + 2) * W - 1, ((i % K) + 1) * W);
    loop_cache_index:
        for (DT j = begin; j < end; j++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline off
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipUram inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false
            if (indexG1IndexPre != (j / K)) indexG1Pre = indexG1Strm.read();
            indexG1IndexPre = (j / K);
            DT indexG1Tmp = indexG1Pre.range((j % K + 1) * W - 1, (j % K) * W);

            //        DT index = indexG1Tmp / K;
            //        DT offset = indexG1Tmp % K;
            DT val1 = indexG1Tmp;
            k00 = val1 % dataOneLine;
            k01 = val1 / dataOneLine;
            k10 = k01 % groupUramPart;
            k11 = k01 / groupUramPart;
            k20 = k11 % uramRow;
            k21 = k11 / uramRow;
            k30 = k21;

            ap_uint<dataOneLine> validTmp;
            if (k01 == validAddrQue[0]) {
                validTmp = validQue[0];
            } else if (k01 == validAddrQue[1]) {
                validTmp = validQue[1];
            } else if (k01 == validAddrQue[2]) {
                validTmp = validQue[2];
            } else if (k01 == validAddrQue[3]) {
                validTmp = validQue[3];
            } else {
                validTmp = valid[k20][k10];
            }
            ap_uint<1> validBool = validTmp[k00]; // to check
            uint512 val2;
            ap_uint<addrWidth> address;
            ap_uint<addrWidth> addrTmp;
            int bramNm = k20 * groupUramPart + k10;
            if (bramNm == addrAddrQue[0]) {
                addrTmp = addrQue[0];
            } else if (bramNm == addrAddrQue[1]) {
                addrTmp = addrQue[1];
            } else if (bramNm == addrAddrQue[2]) {
                addrTmp = addrQue[2];
            } else if (bramNm == addrAddrQue[3]) {
                addrTmp = addrQue[3];
            } else {
                addrTmp = onChipAddr[k20][k10];
            }
            address = addrTmp;

            if (validBool == 0) {
                val2 = offsetG2[k01];
                validTmp = validCnt;
                valid[k20][k10] = validTmp;
                validQue[3] = validQue[2];
                validQue[2] = validQue[1];
                validQue[1] = validQue[0];
                validQue[0] = validTmp;
                validAddrQue[3] = validAddrQue[2];
                validAddrQue[2] = validAddrQue[1];
                validAddrQue[1] = validAddrQue[0];
                validAddrQue[0] = bramNm;
            } else if (address != k30) {
                val2 = offsetG2[k01];
                uint512 val4;
                if (bramNm == ramAddrQue[0]) {
                    val4 = pingQue[0];
                } else if (bramNm == ramAddrQue[1]) {
                    val4 = pingQue[1];
                } else if (bramNm == ramAddrQue[2]) {
                    val4 = pingQue[2];
                } else if (bramNm == ramAddrQue[3]) {
                    val4 = pingQue[3];
                } else {
                    val4 = onChipUram[k20][k10];
                }
                int k01Tmp = (address * uramRow + k20) * groupUramPart + k10;
                offsetG2[k01Tmp] = val4;
            } else {
                if (k01 == ramAddrQue[0]) {
                    val2 = pingQue[0];
                } else if (k01 == ramAddrQue[1]) {
                    val2 = pingQue[1];
                } else if (k01 == ramAddrQue[2]) {
                    val2 = pingQue[2];
                } else if (k01 == ramAddrQue[3]) {
                    val2 = pingQue[3];
                } else {
                    val2 = onChipUram[k20][k10];
                }
            }

            DT dest = val2.range((k00 + 1) * W - 1, W * k00);
            addrG2Strm.write(dest);
            indexG2Strm.write(i);
            dest += 1;
            val2.range((k00 + 1) * W - 1, W * k00) = dest;

            bool flag = ((validBool == 1) && (address == k30));
            if (!flag) {
                ap_uint<addrWidth> tAddr = (ap_uint<addrWidth>)k30;
                addrTmp = tAddr;
                onChipAddr[k20][k10] = addrTmp;
                addrQue[3] = addrQue[2];
                addrQue[2] = addrQue[1];
                addrQue[1] = addrQue[0];
                addrQue[0] = addrTmp;
                addrAddrQue[3] = addrAddrQue[2];
                addrAddrQue[2] = addrAddrQue[1];
                addrAddrQue[1] = addrAddrQue[0];
                addrAddrQue[0] = bramNm;
            }
            onChipUram[k20][k10] = val2;
            pingQue[3] = pingQue[2];
            pingQue[2] = pingQue[1];
            pingQue[1] = pingQue[0];
            pingQue[0] = val2;
            ramAddrQue[3] = ramAddrQue[2];
            ramAddrQue[2] = ramAddrQue[1];
            ramAddrQue[1] = ramAddrQue[0];
            ramAddrQue[0] = bramNm;
        }
        begin = end;
    }
Loop_writeout:
    for (int j = 0; j < uramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
        for (int i = 0; i < groupUramPart; ++i) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16
#pragma HLS pipeline off
            ap_uint<dataOneLine> validTmp;
            validTmp = valid[j][i];
            ap_uint<1> validBool = validTmp[0]; // to check
            if (validBool == 1) {
                ap_uint<addrWidth> address;
                ap_uint<addrWidth> addrTmp;
                int index;
                uint512 val4;
                addrTmp = onChipAddr[j][i];
                address = addrTmp;
                index = (address * uramRow + j) * groupUramPart + i;
                if (index < iteration) {
                    val4 = 0;
                    val4 = onChipUram[j][i];
                    offsetG2[index] = val4;
                }
            }
        }
    }
}

template <typename DT, typename uint512, int rowTemplate, int uramRowBin, int dataOneLineBin, int K, int W, int usURAM>
void calcuIndexWrapper(
    int vertexLen, int edgeLen, uint512* indexG1, uint512* offsetG1, DT* indexG2, uint512* offsetG2) {
#pragma HLS dataflow
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << dataOneLineBin;  // double 8 : float 16
    const int uramRow = 1 << uramRowBin;          // 4096
    const int groupUramPart = 1 << dataUramNmBin; // 8 = 2^3
    const int addrWidth = 32;
    const int unrollFactor = 1;
#ifndef __SYNTHESIS__
    uint512** onChipUram = new uint512*[uramRow];
    ap_uint<addrWidth>** onChipAddr = new ap_uint<addrWidth>*[uramRow];
    ap_uint<dataOneLine>** valid = new ap_uint<dataOneLine>*[uramRow];
    for (int i = 0; i < uramRow; ++i) {
        valid[i] = new ap_uint<dataOneLine>[ groupUramPart ];
        onChipUram[i] = new uint512[groupUramPart];
        onChipAddr[i] = new ap_uint<addrWidth>[ groupUramPart ];
    }
#else
    ap_uint<dataOneLine> valid[uramRow][groupUramPart];
    uint512 onChipUram[uramRow][groupUramPart];
    ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart];
    if (usURAM == 1) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipUram block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUram core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
    } else if (usURAM == 2) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipUram block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUram core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
    } else {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipUram block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUram core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
    }
#endif
    const int strmDepth = 16;
    hls::stream<uint512> offsetG1Strm("offsetG1Strm");
    hls::stream<uint512> indexG1Strm("indexG1Strm");
#pragma HLS stream variable = offsetG1Strm depth = strmDepth
#pragma HLS stream variable = indexG1Strm depth = strmDepth
#pragma HLS resource variable = offsetG1Strm core = FIFO_LUTRAM
#pragma HLS resource variable = indexG1Strm core = FIFO_LUTRAM
    hls::stream<DT> addrStrm;
    hls::stream<DT> indexG2Strm;
#pragma HLS stream variable = addrStrm depth = 256
#pragma HLS stream variable = indexG2Strm depth = 256
#pragma HLS resource variable = addrStrm core = FIFO_LUTRAM
#pragma HLS resource variable = indexG2Strm core = FIFO_LUTRAM
    burstRead2Strm<uint512>((vertexLen + K) / K, offsetG1, offsetG1Strm);
    burstRead2Strm<uint512>((edgeLen + K - 1) / K, indexG1, indexG1Strm);
    calcuIndex<DT, uint512, rowTemplate, uramRow, groupUramPart, dataOneLine, addrWidth, K, W>(
        vertexLen, valid, onChipUram, onChipAddr, indexG1Strm, offsetG1Strm, offsetG2, addrStrm, indexG2Strm);
    writeDDRByAddr<DT>(edgeLen, addrStrm, indexG2Strm, indexG2);
#ifndef __SYNTHESIS__
    for (int i = 0; i < uramRow; ++i) {
        delete[] valid[i];
        delete[] onChipUram[i];
        delete[] onChipAddr[i];
    }
    delete[] onChipAddr;
    delete[] onChipUram;
    delete[] valid;
#endif
}
} // namespace convert_csr_csc
} // namespace internal

/**
 * @brief convert Csr Csc algorithm is implemented
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE number of data in one 512bit in Binary, for double, it's 3, for float, it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 *
 * @param numEdge CSR/CSC data indices number
 * @param numVertex CSR/CSC data offsets number
 * @param indexIn original CSR/CSC data indice array
 * @param offsetIn original CSR/CSC data offset array
 * @param indexOut output transfered CSC/CSR data indice array
 * @param offsetOut output transfered CSC/CSR data offset array
 * @param offsetTmp0 internal temporary CSC/CSR data offset array
 * @param offsetTmp1 internal temporary CSC/CSR data offset array
 */

template <typename DT, int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE>
void convertCsrCsc(int numEdge,
                   int numVertex,
                   ap_uint<512>* offsetIn,
                   ap_uint<512>* indexIn,
                   ap_uint<512>* offsetOut,
                   DT* indexOut,
                   ap_uint<512>* offsetTmp0,
                   ap_uint<512>* offsetTmp1) {
    calcuDegree<MAXVERTEX, MAXEDGE, LOG2CACHEDEPTH, LOG2DATAPERCACHELINE, RAMTYPE>(numVertex, numEdge, indexIn,
                                                                                   offsetTmp0);
    internal::convert_csr_csc::calcuOffset<DT, ap_uint<512>, 16, 32>(numVertex, offsetTmp0, offsetOut, offsetTmp1);
    internal::convert_csr_csc::calcuIndexWrapper<DT, ap_uint<512>, MAXVERTEX, LOG2CACHEDEPTH, LOG2DATAPERCACHELINE, 16,
                                                 32, RAMTYPE>(numVertex, numEdge, indexIn, offsetIn, indexOut,
                                                              offsetTmp1);
}

} // namespace graph
} // namespace xf
#endif
