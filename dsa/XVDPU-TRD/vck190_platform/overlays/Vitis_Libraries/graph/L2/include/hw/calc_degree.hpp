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

#ifndef _XF_GRAPH_CALCULATE_DGREE_HPP_
#define _XF_GRAPH_CALCULATE_DGREE_HPP_

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include <hls_stream.h>
#include "ap_fixed.h"
#include <stdint.h>

namespace xf {
namespace graph {
namespace internal {
namespace calc_degree {

template <typename MType>
union f_cast;

template <>
union f_cast<ap_uint<32> > {
    uint32_t f;
    uint32_t i;
};

template <>
union f_cast<ap_uint<64> > {
    uint64_t f;
    uint64_t i;
};

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
};

#ifndef __SYNTHESIS__
template <typename uint512, int rowTemplate>
void burstRead2Strm(int len, uint512* inArr, hls::stream<uint512>& outStrm) {
#else
template <typename uint512, int rowTemplate>
void burstRead2Strm(int len, uint512 inArr[rowTemplate], hls::stream<uint512>& outStrm) {
#endif
    for (int i = 0; i < len; i++) {
// clang-format off
#pragma HLS loop_tripcount min = 16500000/16 avg = 16500000/16 max = 16500000/16
// clang-format on
#pragma HLS pipeline II = 1
        outStrm.write(inArr[i]);
    }
}

#ifndef __SYNTHESIS__
template <typename uint512, int rowTemplate>
void burstInitDDR(int len, uint512* buf) {
#else
template <typename uint512, int rowTemplate>
void burstInitDDR(int len, uint512 buf[rowTemplate]) {
#endif
    uint512 tmp = 0;
    for (int i = 0; i < len; i++) {
// clang-format off
#pragma HLS loop_tripcount min = 16500000/16 avg = 16500000/16 max = 16500000/16
// clang-format on
#pragma HLS pipeline II = 1
        buf[i] = tmp;
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
void updateDegree(int nrows,
                  int edgeNum,
                  ap_uint<dataOneLine>** valid,
                  ap_uint<512>** onChipUramPing,
                  ap_uint<addrWidth>** onChipAddr,
                  hls::stream<uint512>& indexG1Strm,
                  uint512* offsetG2) {
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
void updateDegree(int nrows,
                  int edgeNum,
                  ap_uint<dataOneLine> valid[uramRow][groupUramPart],
                  ap_uint<512> onChipUramPing[uramRow][groupUramPart],
                  ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart],
                  hls::stream<uint512>& indexG1Strm,
                  uint512 offsetG2[rowTemplate]) {
#endif
    const int iteration = (nrows + 15) / 16;
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
            onChipUramPing[j][i] = 0;
        }
    }

    ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
    ap_uint<512> pingQue[4] = {0, 0, 0, 0};
    ap_uint<512> cntQue[4] = {0, 0, 0, 0};
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
loop_cache:
    for (int i = 0; i < edgeNum; i++) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline off
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipUramPing inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false
        if (i % K == 0) {
            tmp = indexG1Strm.read();
        }
        DT val1 = tmp.range(((i % K) + 1) * W - 1, W * (i % K));
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
                val4 = onChipUramPing[k20][k10];
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
                val2 = onChipUramPing[k20][k10];
            }
        }
        DT val3 = val2.range((k00 + 1) * W - 1, W * k00);
        val3 += 1;
        val2.range((k00 + 1) * W - 1, W * k00) = val3;

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
        onChipUramPing[k20][k10] = val2;
        pingQue[3] = pingQue[2];
        pingQue[2] = pingQue[1];
        pingQue[1] = pingQue[0];
        pingQue[0] = val2;
        ramAddrQue[3] = ramAddrQue[2];
        ramAddrQue[2] = ramAddrQue[1];
        ramAddrQue[1] = ramAddrQue[0];
        ramAddrQue[0] = bramNm;
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
                ap_uint<512> val4;
                addrTmp = onChipAddr[j][i];
                address = addrTmp;
                index = (address * uramRow + j) * groupUramPart + i;
                if (index < iteration) {
                    val4 = 0;
                    val4 = onChipUramPing[j][i];
                    offsetG2[index] = val4;
                }
            }
        }
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
void updateDegreeWeighted(int nrows,
                          int edgeNum,
                          ap_uint<dataOneLine>** valid,
                          ap_uint<512>** onChipUramPing,
                          ap_uint<addrWidth>** onChipAddr,
                          hls::stream<uint512>& indexG1Strm,
                          hls::stream<uint512>& weightStrm,
                          uint512* offsetG2) {
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
void updateDegreeWeighted(int nrows,
                          int edgeNum,
                          ap_uint<dataOneLine> valid[uramRow][groupUramPart],
                          ap_uint<512> onChipUramPing[uramRow][groupUramPart],
                          ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart],
                          hls::stream<uint512>& indexG1Strm,
                          hls::stream<uint512>& weightStrm,
                          uint512 offsetG2[rowTemplate]) {
#endif
    const int widthT = sizeof(DT) * 8;
    const int iteration = (nrows + 15) / 16;
    ap_uint<dataOneLine> validCnt = -1;
    uint512 tmp = 0;
    uint512 wei512 = 0;
    f_cast<DT> init;
    init.f = 0.0;
Loop_init_uram:
    for (int j = 0; j < uramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
        for (int i = 0; i < groupUramPart; ++i) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16
#pragma HLS pipeline II = 1
            valid[j][i] = 0;
            onChipAddr[j][i] = 0;
            onChipUramPing[j][i] = init.i;
        }
    }

    ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
    ap_uint<512> pingQue[4] = {0, 0, 0, 0};
    ap_uint<512> cntQue[4] = {0, 0, 0, 0};
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

    ap_uint<32> k00, k01, k10, k11, k20, k21, k30;
loop_cache:
    for (int i = 0; i < edgeNum; i++) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline off
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipUramPing inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false
        if (i % K == 0) {
            tmp = indexG1Strm.read();
            wei512 = weightStrm.read();
        }
        // get index by ap_uint<32> format
        ap_uint<32> val1 = tmp.range(((i % K) + 1) * W - 1, W * (i % K));
        // cast weight by DT format
        f_cast<DT> weight_cast;
        weight_cast.i = wei512.range(((i % K) + 1) * W - 1, W * (i % K));
        DT weight = weight_cast.f;

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
                val4 = onChipUramPing[k20][k10];
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
                val2 = onChipUramPing[k20][k10];
            }
        }

        f_cast<DT> val_cast;
        val_cast.i = val2.range((k00 + 1) * W - 1, W * k00);

#ifndef __SYNTHESIS__
// std::cout << "val_cast.f : " << val_cast.f << std::endl;
#endif

        val_cast.f += weight;

        val2.range((k00 + 1) * W - 1, W * k00) = val_cast.i; // the degree val

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
        onChipUramPing[k20][k10] = val2;
        pingQue[3] = pingQue[2];
        pingQue[2] = pingQue[1];
        pingQue[1] = pingQue[0];
        pingQue[0] = val2;
        ramAddrQue[3] = ramAddrQue[2];
        ramAddrQue[2] = ramAddrQue[1];
        ramAddrQue[1] = ramAddrQue[0];
        ramAddrQue[0] = bramNm;
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
                ap_uint<512> val4;
                addrTmp = onChipAddr[j][i];
                address = addrTmp;
                index = (address * uramRow + j) * groupUramPart + i;
                if (index < iteration) {
                    val4 = 0;
                    val4 = onChipUramPing[j][i];
                    offsetG2[index] = val4;
                }
            }
        }
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
          int unrollBin,
          int CHANNELNUM,
          int K = 16,
          int W = 32>
void updateDegreeWeighted(int nrows,
                          int edgeNum,
                          int numEdgePerChannel[CHANNELNUM],
                          ap_uint<dataOneLine>** valid,
                          ap_uint<512>** onChipUramPing,
                          ap_uint<addrWidth>** onChipAddr,
                          hls::stream<uint512>& indexG1Strm,
                          hls::stream<uint512>& weightStrm,
                          uint512* offsetG2) {
#else
template <typename DT,
          typename uint512,
          int rowTemplate,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,
          int unrollBin,
          int CHANNELNUM,
          int K = 16,
          int W = 32>
void updateDegreeWeighted(int nrows,
                          int edgeNum,
                          int numEdgePerChannel[CHANNELNUM],
                          ap_uint<dataOneLine> valid[uramRow][groupUramPart],
                          ap_uint<512> onChipUramPing[uramRow][groupUramPart],
                          ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart],
                          hls::stream<uint512>& indexG1Strm,
                          hls::stream<uint512>& weightStrm,
                          uint512 offsetG2[rowTemplate]) {
#endif
    const int widthT = sizeof(DT) * 8;
    const int iteration = (nrows + 15) / 16;
    ap_uint<dataOneLine> validCnt = -1;
    uint512 tmp = 0;
    uint512 wei512 = 0;
    f_cast<DT> init;
    init.f = 0.0;
Loop_init_uram:
    for (int j = 0; j < uramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
        for (int i = 0; i < groupUramPart; ++i) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16
#pragma HLS pipeline II = 1
            valid[j][i] = 0;
            onChipAddr[j][i] = 0;
            onChipUramPing[j][i] = init.i;
        }
    }
    for (int n = 0; n < CHANNELNUM; n++) {
#pragma HLS UNROLL
        numEdgePerChannel[n] = 0;
    }

    ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
    ap_uint<512> pingQue[4] = {0, 0, 0, 0};
    ap_uint<512> cntQue[4] = {0, 0, 0, 0};
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

    ap_uint<32> k00, k01, k10, k11, k20, k21, k30;
loop_cache:
    for (int i = 0; i < edgeNum; i++) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline off
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipUramPing inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false
        if (i % K == 0) {
            tmp = indexG1Strm.read();
            wei512 = weightStrm.read();
        }
        // get index by ap_uint<32> format
        ap_uint<32> val1 = tmp.range(((i % K) + 1) * W - 1, W * (i % K));
        ap_uint<32> Channel = val1 >> unrollBin;

        for (int n = 0; n < CHANNELNUM; n++) {
#pragma HLS UNROLL
            if (CHANNELNUM == 2) {
                if (Channel[0] == n) {
                    numEdgePerChannel[n]++;
                }
            } else if (CHANNELNUM == 8) {
                if (Channel(2, 0) == n) {
                    numEdgePerChannel[n]++;
                }
            } else if (CHANNELNUM == 6) {
                if (Channel % 6 == n) {
                    numEdgePerChannel[n]++;
                }
            }
        }
        // cast weight by DT format
        f_cast<DT> weight_cast;
        weight_cast.i = wei512.range(((i % K) + 1) * W - 1, W * (i % K));
        DT weight = weight_cast.f;

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
                val4 = onChipUramPing[k20][k10];
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
                val2 = onChipUramPing[k20][k10];
            }
        }

        f_cast<DT> val_cast;
        val_cast.i = val2.range((k00 + 1) * W - 1, W * k00);

#ifndef __SYNTHESIS__
// std::cout << "val_cast.f : " << val_cast.f << std::endl;
#endif

        val_cast.f += weight;

        val2.range((k00 + 1) * W - 1, W * k00) = val_cast.i; // the degree val

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
        onChipUramPing[k20][k10] = val2;
        pingQue[3] = pingQue[2];
        pingQue[2] = pingQue[1];
        pingQue[1] = pingQue[0];
        pingQue[0] = val2;
        ramAddrQue[3] = ramAddrQue[2];
        ramAddrQue[2] = ramAddrQue[1];
        ramAddrQue[1] = ramAddrQue[0];
        ramAddrQue[0] = bramNm;
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
                ap_uint<512> val4;
                addrTmp = onChipAddr[j][i];
                address = addrTmp;
                index = (address * uramRow + j) * groupUramPart + i;
                if (index < iteration) {
                    val4 = 0;
                    val4 = onChipUramPing[j][i];
                    offsetG2[index] = val4;
                }
            }
        }
    }
}

} // namespace calc_degree
} // namespace internal

/**
 * @brief calculate degree algorithm is implemented
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE number of data in one 512bit in Binary, for double, it's 3, for float, it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 *
 * @param numVertex CSR/CSC data vertex number
 * @param numEdge CSR/CSC data edge number
 * @param index input CSR/CSC data index array
 * @param degree output degree array
 */

#ifndef __SYNTHESIS__
template <int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE>
void calcuDegree(int numVertex, int numEdge, ap_uint<512>* index, ap_uint<512>* degree) {
#else
template <int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE>
void calcuDegree(int numVertex, int numEdge, ap_uint<512> index[MAXEDGE], ap_uint<512> degree[MAXVERTEX]) {
#endif
#pragma HLS dataflow
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << LOG2DATAPERCACHELINE; // double 8 : float 16
    const int uramRow = 1 << LOG2CACHEDEPTH;           // 4096
    const int groupUramPart = 1 << dataUramNmBin;      // 8 = 2^3
    const int addrWidth = 32;
    const int unrollFactor = 1;
#ifndef __SYNTHESIS__
    ap_uint<512>** onChipUramPing = new ap_uint<512>*[uramRow];
    ap_uint<addrWidth>** onChipAddr = new ap_uint<addrWidth>*[uramRow];
    ap_uint<dataOneLine>** valid = new ap_uint<dataOneLine>*[uramRow];
    for (int i = 0; i < uramRow; ++i) {
        valid[i] = new ap_uint<dataOneLine>[ groupUramPart ];
        onChipUramPing[i] = new ap_uint<512>[ groupUramPart ];
        onChipAddr[i] = new ap_uint<addrWidth>[ groupUramPart ];
    }
#else
    ap_uint<dataOneLine> valid[uramRow][groupUramPart];
    ap_uint<512> onChipUramPing[uramRow][groupUramPart];
    ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart];
    if (RAMTYPE == 1) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
    } else if (RAMTYPE == 2) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
    } else {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
    }
#endif
    hls::stream<ap_uint<512> > indexG1Strm("indexG1Strm");
#pragma HLS stream variable = indexG1Strm depth = 16
#pragma HLS resource variable = indexG1Strm core = FIFO_LUTRAM
    internal::calc_degree::burstRead2Strm<ap_uint<512>, MAXEDGE>((numEdge + 16 - 1) / 16, index, indexG1Strm);
    internal::calc_degree::updateDegree<ap_uint<32>, ap_uint<512>, MAXVERTEX, uramRow, groupUramPart, dataOneLine,
                                        addrWidth, 16, 32>(numVertex, numEdge, valid, onChipUramPing, onChipAddr,
                                                           indexG1Strm, degree);
#ifndef __SYNTHESIS__
    for (int i = 0; i < uramRow; ++i) {
        delete[] valid[i];
        delete[] onChipUramPing[i];
        delete[] onChipAddr[i];
    }
    delete[] onChipAddr;
    delete[] onChipUramPing;
    delete[] valid;
#endif
}

/**
 * @brief calculate weighted degree algorithm is implemented
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE number of data in one 512bit in Binary, for double, it's 3, for float, it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 *
 * @param numVertex CSR/CSC data vertex number
 * @param numEdge CSR/CSC data edge number
 * @param index input CSR/CSC data index array
 * @param weight input CSR/CSC data weight array, default float type.
 * @param degree output degree array, default float type.
 */

template <int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE>
void calcuWeightedDegree(int numVertex,
                         int numEdge,
                         ap_uint<512> index[MAXEDGE],
                         ap_uint<512> weight[MAXEDGE],
                         ap_uint<512> degree[MAXVERTEX]) {
#pragma HLS dataflow
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << LOG2DATAPERCACHELINE; // double 8 : float 16
    const int uramRow = 1 << LOG2CACHEDEPTH;           // 4096
    const int groupUramPart = 1 << dataUramNmBin;      // 8 = 2^3
    const int addrWidth = 32;
    const int unrollFactor = 1;
#ifndef __SYNTHESIS__
    ap_uint<512>** onChipUramPing = new ap_uint<512>*[uramRow];
    ap_uint<addrWidth>** onChipAddr = new ap_uint<addrWidth>*[uramRow];
    ap_uint<dataOneLine>** valid = new ap_uint<dataOneLine>*[uramRow];
    for (int i = 0; i < uramRow; ++i) {
        valid[i] = new ap_uint<dataOneLine>[ groupUramPart ];
        onChipUramPing[i] = new ap_uint<512>[ groupUramPart ];
        onChipAddr[i] = new ap_uint<addrWidth>[ groupUramPart ];
    }
#else
    ap_uint<dataOneLine> valid[uramRow][groupUramPart];
    ap_uint<512> onChipUramPing[uramRow][groupUramPart];
    ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart];
    if (RAMTYPE == 1) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
    } else if (RAMTYPE == 2) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
    } else {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
    }
#endif
    hls::stream<ap_uint<512> > indexG1Strm("indexG1Strm");
#pragma HLS stream variable = indexG1Strm depth = 16
#pragma HLS resource variable = indexG1Strm core = FIFO_LUTRAM
    hls::stream<ap_uint<512> > weightStrm("weightStrm");
#pragma HLS stream variable = weightStrm depth = 16
#pragma HLS resource variable = weightStrm core = FIFO_LUTRAM
    internal::calc_degree::burstRead2Strm<ap_uint<512>, MAXEDGE>((numEdge + 16 - 1) / 16, index, indexG1Strm);
    internal::calc_degree::burstRead2Strm<ap_uint<512>, MAXEDGE>((numEdge + 16 - 1) / 16, weight, weightStrm);
    internal::calc_degree::updateDegreeWeighted<float, ap_uint<512>, MAXVERTEX, uramRow, groupUramPart, dataOneLine,
                                                addrWidth, 16, 32>(numVertex, numEdge, valid, onChipUramPing,
                                                                   onChipAddr, indexG1Strm, weightStrm, degree);
#ifndef __SYNTHESIS__
    for (int i = 0; i < uramRow; ++i) {
        delete[] valid[i];
        delete[] onChipUramPing[i];
        delete[] onChipAddr[i];
    }
    delete[] onChipAddr;
    delete[] onChipUramPing;
    delete[] valid;
#endif
}

/**
 * @brief calculate weighted degree algorithm is implemented
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE number of data in one 512bit in Binary, for double, it's 3, for float, it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 *
 * @param numVertex CSR/CSC data vertex number
 * @param numEdge CSR/CSC data edge number
 * @param index input CSR/CSC data index array
 * @param weight input CSR/CSC data weight array, default float type.
 * @param degree output degree array, default float type.
 */

template <int MAXVERTEX,
          int MAXEDGE,
          int LOG2CACHEDEPTH,
          int LOG2DATAPERCACHELINE,
          int RAMTYPE,
          int unrollbin,
          int CHANNELNUM>
void calcuWeightedDegree(int numVertex,
                         int numEdge,
                         int numEdgePerChannel[CHANNELNUM],
                         ap_uint<512> index[MAXEDGE],
                         ap_uint<512> weight[MAXEDGE],
                         ap_uint<512> degree[MAXVERTEX]) {
#pragma HLS dataflow
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << LOG2DATAPERCACHELINE; // double 8 : float 16
    const int uramRow = 1 << LOG2CACHEDEPTH;           // 4096
    const int groupUramPart = 1 << dataUramNmBin;      // 8 = 2^3
    const int addrWidth = 32;
    const int unrollFactor = 1;
#ifndef __SYNTHESIS__
    ap_uint<512>** onChipUramPing = new ap_uint<512>*[uramRow];
    ap_uint<addrWidth>** onChipAddr = new ap_uint<addrWidth>*[uramRow];
    ap_uint<dataOneLine>** valid = new ap_uint<dataOneLine>*[uramRow];
    for (int i = 0; i < uramRow; ++i) {
        valid[i] = new ap_uint<dataOneLine>[ groupUramPart ];
        onChipUramPing[i] = new ap_uint<512>[ groupUramPart ];
        onChipAddr[i] = new ap_uint<addrWidth>[ groupUramPart ];
    }
#else
    ap_uint<dataOneLine> valid[uramRow][groupUramPart];
    ap_uint<512> onChipUramPing[uramRow][groupUramPart];
    ap_uint<addrWidth> onChipAddr[uramRow][groupUramPart];
    if (RAMTYPE == 1) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_URAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
    } else if (RAMTYPE == 2) {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_BRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
    } else {
#pragma HLS array_partition variable = valid block factor = unrollFactor dim = 2
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipUramPing block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipUramPing core = RAM_S2P_LUTRAM
#pragma HLS array_partition variable = onChipAddr block factor = unrollFactor dim = 2
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
    }
#endif
    hls::stream<ap_uint<512> > indexG1Strm("indexG1Strm");
#pragma HLS stream variable = indexG1Strm depth = 16
#pragma HLS resource variable = indexG1Strm core = FIFO_LUTRAM
    hls::stream<ap_uint<512> > weightStrm("weightStrm");
#pragma HLS stream variable = weightStrm depth = 16
#pragma HLS resource variable = weightStrm core = FIFO_LUTRAM
    internal::calc_degree::burstRead2Strm<ap_uint<512>, MAXEDGE>((numEdge + 16 - 1) / 16, index, indexG1Strm);
    internal::calc_degree::burstRead2Strm<ap_uint<512>, MAXEDGE>((numEdge + 16 - 1) / 16, weight, weightStrm);
    internal::calc_degree::updateDegreeWeighted<float, ap_uint<512>, MAXVERTEX, uramRow, groupUramPart, dataOneLine,
                                                addrWidth, unrollbin, CHANNELNUM, 16, 32>(
        numVertex, numEdge, numEdgePerChannel, valid, onChipUramPing, onChipAddr, indexG1Strm, weightStrm, degree);
#ifndef __SYNTHESIS__
    for (int i = 0; i < uramRow; ++i) {
        delete[] valid[i];
        delete[] onChipUramPing[i];
        delete[] onChipAddr[i];
    }
    delete[] onChipAddr;
    delete[] onChipUramPing;
    delete[] valid;
#endif
}

} // namespace graph
} // namespace xf
#endif
