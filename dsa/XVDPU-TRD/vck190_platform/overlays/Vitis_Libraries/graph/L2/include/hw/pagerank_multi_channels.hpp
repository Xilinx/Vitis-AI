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
 * @file pagerank.hpp
 * @brief  This files contains implementation of PageRank
 */

#ifndef XF_GRAPH_PR_MULTI_H
#define XF_GRAPH_PR_MULTI_H

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "hls_math.h"
#include <hls_stream.h>

#include "calc_degree.hpp"
#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/cache.hpp"

#define _Width 512
typedef ap_uint<_Width> buffT;
// user define the Channel numbers for PageRank, support 2 or 6
// define N data channels using HBM, Support 2, 6 channels
#define CHANNEL_NUM (2)

namespace xf {
namespace graph {
namespace internal {
namespace pagerankMultiChannel {

/*
 * brief cache is a URAM design for caching Read-only DDR/HBM memory spaces
 *
 * This function stores history data recently loaded from DDR/HBM in the on-chip memory(URAM).
 * It aims to reduce DDR/HBM access when the memory is accessed randomly.
 *
 * tparam T The type of the actual data accessed. Float and double is not supported.
 * tparam ramRow			The number of rows each on chip ram has
 * tparam groupramPart		The number of on chip ram used in cache
 * tparam dataOneLine		The number of actual data each 512 can contain
 * tparam addrWidth		The width of the address to access the memory
 * tparam validRamType     The ram type of the valid flag array. 0 for LUTRAM,
 * 1 for BRAM, 2 for URAM
 * tparam addrRamType      The ram type of the onchip addr array. 0 for LUTRAM,
 * 1 for BRAM, 2 for URAM
 * tparam dataRamType      The ram type of the onchip data array. 0 for LUTRAM,
 * 1 for BRAM, 2 for URAM
 */

template <typename T,
          int ramRow,
          int groupRamPart,
          int dataOneLine,
          int addrWidth,
          int validRamType,
          int addrRamType,
          int dataRamType>
class cache {
   public:
    cache() {
#pragma HLS inline
#pragma HLS array_partition variable = valid block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipRam1 block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipRam0 block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipAddr block factor = 1 dim = 2

        if (validRamType == 0) {
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
        } else if (validRamType == 2) {
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = valid core = RAM_S2P_URAM
        }

        if (addrRamType == 0) {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
        } else if (addrRamType == 2) {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
        }

        if (dataRamType == 0) {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_LUTRAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_LUTRAM
        } else if (dataRamType == 2) {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_BRAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_URAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_URAM
        }

#ifndef __SYNTHESIS__
        valid = new ap_uint<dataOneLine>*[ramRow];
        onChipRam1 = new ap_uint<512>*[ramRow];
        onChipRam0 = new ap_uint<512>*[ramRow];
        onChipAddr = new ap_uint<addrWidth>*[ramRow];
        for (int i = 0; i < ramRow; ++i) {
            valid[i] = new ap_uint<dataOneLine>[ groupRamPart ];
            onChipRam0[i] = new ap_uint<512>[ groupRamPart ];
            onChipRam1[i] = new ap_uint<512>[ groupRamPart ];
            onChipAddr[i] = new ap_uint<addrWidth>[ groupRamPart ];
        }
#endif
    }

    ~cache() {
#ifndef __SYNTHESIS__
        for (int i = 0; i < ramRow; ++i) {
            delete[] valid[i];
            delete[] onChipRam0[i];
            delete[] onChipRam1[i];
            delete[] onChipAddr[i];
        }
        delete[] onChipAddr;
        delete[] onChipRam0;
        delete[] onChipRam1;
        delete[] valid;
#endif
    }

    // Initialization of the on chip memory when controlling single off
    // chip memory.
    void initSingleOffChip() {
    Loop_init_uram:
        for (int j = 0; j < ramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
            for (int i = 0; i < groupRamPart; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
                valid[j][i] = 0;
                onChipAddr[j][i] = 0;
                onChipRam0[j][i] = 0;
            }
        }
    }

    // brief  Initialization of the on chip memory when controlling dual off
    // chip memory.
    void initDualOffChip() {
    Loop_init_uram:
        for (int j = 0; j < ramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
            for (int i = 0; i < groupRamPart; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
                valid[j][i] = 0;
                onChipAddr[j][i] = 0;
                onChipRam0[j][i] = 0;
                onChipRam1[j][i] = 0;
            }
        }
    }
    // brief readOnly function that index two off-chip buffers without end
    // flags. Both of the buffers should be
    // indexed in exactly the same behaviour.
    // param ddrMem0		The pointer for the first off chip memory
    // param ddrMem1		The pointer for the second off chip memory
    // param addrStrm		The read address should be sent from this stream
    // param e_addrStrm   The end flag stream of the addrStrm
    // param data0Strm	The data loaded from the first off chip memory
    // param data1Strm	The data loaded from the second off chip memory
    // param e_dataStrm   The end flag stream of the synchronous dataStrm
    void readOnly(int cnt,
                  ap_uint<512>* ddrMem0,
                  ap_uint<512>* ddrMem1,
                  ap_uint<4> channelNum,
                  hls::stream<ap_uint<32> >& addrStrm,
                  hls::stream<ap_uint<32> >& bypassStrm,

                  hls::stream<T>& data0Strm,
                  hls::stream<T>& data1Strm,
                  hls::stream<ap_uint<32> >& passStrm) {
#pragma HLS inline off

        ap_uint<dataOneLine> validCnt;
        for (int i = 0; i < dataOneLine; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
            validCnt[i] = 1;
        }
        const int size = sizeof(T) * 8;

        ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
        ap_uint<512> pingQue[4] = {0, 0, 0, 0};
        ap_uint<512> cntQue[4] = {0, 0, 0, 0};
        ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
        int validAddrQue[4] = {-1, -1, -1, -1};
        int addrAddrQue[4] = {-1, -1, -1, -1};
        int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = cntQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

        ap_uint<32> channel = channelNum;
        int count = 0;

        while (count < cnt) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipRam0 inter false
#pragma HLS DEPENDENCE variable = onChipRam1 inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false

            if (addrStrm.empty() == false) {
                ap_uint<32> bypass = bypassStrm.read();
                int addrNchannel = addrStrm.read();

                int index;
#if (CHANNEL_NUM == 6)
                if (dataOneLine == 16) {
                    index = addrNchannel - ((addrNchannel / 96) * 80) - (channel << 4);
                } else {
                    index = addrNchannel - ((addrNchannel / 48) * 40) - (channel << 3);
                }
#else
                if (dataOneLine == 16) {
                    index = addrNchannel - ((addrNchannel >> 5) << 4) - (channel << 4);
                } else {
                    index = addrNchannel - ((addrNchannel >> 4) << 3) - (channel << 3);
                }
#endif
                int k00 = index % dataOneLine;
                int k01 = index / dataOneLine;
                int k10 = k01 % groupRamPart;
                int k11 = k01 / groupRamPart;
                int k20 = k11 % ramRow;
                int k21 = k11 / ramRow;
                int k30 = k21;

#ifndef __SYNTHESIS__
// std::cout<<"index : "<<index<<" : "<<addrNchannel<<std::endl;
#endif

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
                ap_uint<512> tmpV, tmpC;
                ap_uint<addrWidth> address;
                ap_uint<addrWidth> addrTmp;
                if (k01 == addrAddrQue[0]) {
                    addrTmp = addrQue[0];
                } else if (k01 == addrAddrQue[1]) {
                    addrTmp = addrQue[1];
                } else if (k01 == addrAddrQue[2]) {
                    addrTmp = addrQue[2];
                } else if (k01 == addrAddrQue[3]) {
                    addrTmp = addrQue[3];
                } else {
                    addrTmp = onChipAddr[k20][k10];
                }
                address = addrTmp;
                if ((validBool == 1) && (address == k30)) {
                    if (k01 == ramAddrQue[0]) {
                        tmpV = pingQue[0];
                        tmpC = cntQue[0];
                    } else if (k01 == ramAddrQue[1]) {
                        tmpV = pingQue[1];
                        tmpC = cntQue[1];
                    } else if (k01 == ramAddrQue[2]) {
                        tmpV = pingQue[2];
                        tmpC = cntQue[2];
                    } else if (k01 == ramAddrQue[3]) {
                        tmpV = pingQue[3];
                        tmpC = cntQue[3];
                    } else {
                        tmpV = onChipRam0[k20][k10];
                        tmpC = onChipRam1[k20][k10];
                    }
                } else {
                    tmpV = ddrMem0[k01];
                    tmpC = ddrMem1[k01];
                    onChipRam0[k20][k10] = tmpV;
                    onChipRam1[k20][k10] = tmpC;
                    pingQue[3] = pingQue[2];
                    pingQue[2] = pingQue[1];
                    pingQue[1] = pingQue[0];
                    pingQue[0] = tmpV;
                    cntQue[3] = cntQue[2];
                    cntQue[2] = cntQue[1];
                    cntQue[1] = cntQue[0];
                    cntQue[0] = tmpC;
                    ramAddrQue[3] = ramAddrQue[2];
                    ramAddrQue[2] = ramAddrQue[1];
                    ramAddrQue[1] = ramAddrQue[0];
                    ramAddrQue[0] = k01;
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
                    addrAddrQue[0] = k01;
                }
                if (validBool == 0) {
                    validTmp = validCnt;
                    valid[k20][k10] = validTmp;
                    validQue[3] = validQue[2];
                    validQue[2] = validQue[1];
                    validQue[1] = validQue[0];
                    validQue[0] = validTmp;
                    validAddrQue[3] = validAddrQue[2];
                    validAddrQue[2] = validAddrQue[1];
                    validAddrQue[1] = validAddrQue[0];
                    validAddrQue[0] = k01;
                }
                data0Strm.write(tmpV.range(size * (k00 + 1) - 1, size * k00));
                data1Strm.write(tmpC.range(size * (k00 + 1) - 1, size * k00));
                passStrm.write(bypass);
                count++;

            } // end if
        }     // end while
    }

   private:
#ifndef __SYNTHESIS__
    ap_uint<dataOneLine>** valid;
    ap_uint<512>** onChipRam0;
    ap_uint<512>** onChipRam1;
    ap_uint<addrWidth>** onChipAddr;
#else
    ap_uint<dataOneLine> valid[ramRow][groupRamPart];
    ap_uint<512> onChipRam0[ramRow][groupRamPart];
    ap_uint<512> onChipRam1[ramRow][groupRamPart];
    ap_uint<addrWidth> onChipAddr[ramRow][groupRamPart];
#endif

}; // cache class

// clang-format off
template <typename T, int rowTemplate, int UN>
void convergenceSub(int nrows, T tol, hls::stream<buffT>& pingStrm, hls::stream<buffT>& pongStrm, bool& converged) {
// clang-format on
#pragma HLS inline off
    const int size = sizeof(T) * 8;
    const int wN = UN;
    const int iteration = (nrows + wN - 1) / wN;
    converged = 1;

    bool converg[UN];
    for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS unroll factor = wN
        converg[k] = 1;
    }

    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        buffT tmpA, tmpB;
        tmpA = pingStrm.read();
        tmpB = pongStrm.read();
        for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS unroll factor = wN
            int index = i * wN + k;
            if (index < nrows) {
                calc_degree::f_cast<T> tmp1;
                calc_degree::f_cast<T> tmp2;
                tmp1.i = tmpA.range(size * (k + 1) - 1, size * k);
                tmp2.i = tmpB.range(size * (k + 1) - 1, size * k);
                T tmpVal = hls::abs(tmp1.f - tmp2.f);
                if (tmpVal > tol) {
                    converg[k] = 0;
                }
            }
        }
    }

    for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS unroll factor = wN
        if (converg[k] == 0) {
            converged = 0;
        }
    }
}

template <typename T, int rowTemplate>
void calConvergence2channels(int nLinePerChannel0,
                             int nLinePerChannel1,
                             int nReadChannel0,
                             int nReadChannel1,
                             T tol,
                             bool& channelConverged0,
                             buffT bufferPing0[rowTemplate],
                             buffT bufferPong0[rowTemplate],
                             bool& channelConverged1,
                             buffT bufferPing1[rowTemplate],
                             buffT bufferPong1[rowTemplate]) {
#pragma HLS inline off
#pragma HLS dataflow

    const int wN = (sizeof(T) == 4) ? 16 : 8;

    hls::stream<buffT> pingStrm[2];
    hls::stream<buffT> pongStrm[2];
// clang-format on
#pragma HLS stream depth = 32 variable = pingStrm
#pragma HLS stream depth = 32 variable = pongStrm
#pragma HLS resource variable = pingStrm core = FIFO_LUTRAM
#pragma HLS resource variable = pongStrm core = FIFO_LUTRAM
    xf::graph::internal::burstRead2Strm<buffT>(nLinePerChannel0, bufferPing0, pingStrm[0]);
    xf::graph::internal::burstRead2Strm<buffT>(nLinePerChannel1, bufferPing1, pingStrm[1]);
    xf::graph::internal::burstRead2Strm<buffT>(nLinePerChannel0, bufferPong0, pongStrm[0]);
    xf::graph::internal::burstRead2Strm<buffT>(nLinePerChannel1, bufferPong1, pongStrm[1]);
    convergenceSub<T, rowTemplate, wN>(nReadChannel0, tol, pingStrm[0], pongStrm[0], channelConverged0);
    convergenceSub<T, rowTemplate, wN>(nReadChannel1, tol, pingStrm[1], pongStrm[1], channelConverged1);
}
template <typename T, int rowTemplate>
void calConvergence(int nrows,
                    T tol,
                    bool& channelConverged0,
                    buffT bufferPing0[rowTemplate],
                    buffT bufferPong0[rowTemplate],
                    bool& channelConverged1,
                    buffT bufferPing1[rowTemplate],
                    buffT bufferPong1[rowTemplate]
#if (CHANNEL_NUM == 6)
                    ,
                    bool& channelConverged2,
                    buffT bufferPing2[rowTemplate],
                    buffT bufferPong2[rowTemplate],
                    bool& channelConverged3,
                    buffT bufferPing3[rowTemplate],
                    buffT bufferPong3[rowTemplate],
                    bool& channelConverged4,
                    buffT bufferPing4[rowTemplate],
                    buffT bufferPong4[rowTemplate],
                    bool& channelConverged5,
                    buffT bufferPing5[rowTemplate],
                    buffT bufferPong5[rowTemplate]
#endif
                    ) {

#pragma HLS inline off

    const int wN = (sizeof(T) == 4) ? 16 : 8;
    const int BURST_LENTH = 32;
    const int iteration = (nrows + wN - 1) / wN; // total lines
    const int cacheOneLineBin = (sizeof(T) == 4) ? 4 : 3;
    ap_uint<32> nAvgReadPerChannel = iteration / CHANNEL_NUM; // avg lines per channel
    int nextraRead = iteration - nAvgReadPerChannel * CHANNEL_NUM;

    int extra = nrows - (iteration - 1) * wN;
    int nReadChannel[CHANNEL_NUM];
    int nLinePerChannel[CHANNEL_NUM];

#if (CHANNEL_NUM == 6)
    if (nextraRead > 0) {
        for (int i = 0; i < CHANNEL_NUM; ++i) {
#pragma HLS pipeline II = 1
            if (i < nextraRead - 1) {
                nReadChannel[i] = (nAvgReadPerChannel + 1) << cacheOneLineBin;
                nLinePerChannel[i] = nAvgReadPerChannel + 1;
            } else if (i == nextraRead - 1) {
                nReadChannel[i] = (nAvgReadPerChannel << cacheOneLineBin) + extra;
                nLinePerChannel[i] = nAvgReadPerChannel + 1;
            } else {
                nReadChannel[i] = ((nAvgReadPerChannel) << cacheOneLineBin);
                nLinePerChannel[i] = nAvgReadPerChannel;
            }
        } // end unroll
    } else {
        for (int i = 0; i < CHANNEL_NUM - 1; ++i) {
#pragma HLS pipeline II = 1
            nReadChannel[i] = nAvgReadPerChannel << cacheOneLineBin;
            nLinePerChannel[i] = nAvgReadPerChannel;
        }
        nReadChannel[CHANNEL_NUM - 1] = ((nAvgReadPerChannel - 1) << cacheOneLineBin) + extra;
        nLinePerChannel[CHANNEL_NUM - 1] = nAvgReadPerChannel;
    }
#else
    if (nextraRead == 1) {
        nReadChannel[0] = (nAvgReadPerChannel << cacheOneLineBin) + extra;
        nReadChannel[1] = nAvgReadPerChannel << cacheOneLineBin;
        nLinePerChannel[0] = nAvgReadPerChannel + 1;
        nLinePerChannel[1] = nAvgReadPerChannel;
    } else {
        nReadChannel[0] = nAvgReadPerChannel << cacheOneLineBin;
        nReadChannel[1] = ((nAvgReadPerChannel - 1) << cacheOneLineBin) + extra;
        nLinePerChannel[0] = nAvgReadPerChannel;
        nLinePerChannel[1] = nAvgReadPerChannel;
    }
#endif

#ifndef __SYNTHESIS__
    std::cout << "iteration : " << iteration << ":" << nAvgReadPerChannel << ":" << extra << std::endl;
    for (int i = 0; i < CHANNEL_NUM; ++i) {
        std::cout << "nReadChannel/Line[i] : " << nReadChannel[i] << ":" << nLinePerChannel[i] << std::endl;
    }
#endif

    // Serial execution to save resources
    for (int i = 0; i < (CHANNEL_NUM / 2); ++i) {
        if (i == 0) {
            calConvergence2channels<T, rowTemplate>(nLinePerChannel[0], nLinePerChannel[1], nReadChannel[0],
                                                    nReadChannel[1], tol, channelConverged0, bufferPing0, bufferPong0,
                                                    channelConverged1, bufferPing1, bufferPong1);
#if (CHANNEL_NUM == 6)
        } else if (i == 1) {
            calConvergence2channels<T, rowTemplate>(
                nLinePerChannel[2 * i], nLinePerChannel[2 * i + 1], nReadChannel[2 * i], nReadChannel[2 * i + 1], tol,
                channelConverged2, bufferPing2, bufferPong2, channelConverged3, bufferPing3, bufferPong3);
        } else if (i == 2) {
            calConvergence2channels<T, rowTemplate>(
                nLinePerChannel[2 * i], nLinePerChannel[2 * i + 1], nReadChannel[2 * i], nReadChannel[2 * i + 1], tol,
                channelConverged4, bufferPing4, bufferPong4, channelConverged5, bufferPing5, bufferPong5);
#endif
        }
    }
}

// clang-format off
template <typename T, int unrollNm, int unrollBin>
void collect(
                 int nnz,
                 hls::stream<T> mulStrm[CHANNEL_NUM],
				 hls::stream<ap_uint<4 > >&	channelStrm,
                 hls::stream<T>& mulCollStrm
){
// clang-format on
#pragma HLS inline off

    int channelNum = channelStrm.read();
    for (int i = 0; i < nnz; ++i) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
        T tmp = mulStrm[channelNum].read();
        mulCollStrm.write(tmp);
        channelNum = channelStrm.read();
#ifndef __SYNTHESIS__
        if (20007 == i) std::cout << "vertex 1 tmp: " << tmp << std::endl;
#endif
    }
}

template <int _W, int unrollBin, int _NStrm>
void dispatchIndice(int nnz,
                    hls::stream<ap_uint<_W> >& indiceFixedStrm,
                    hls::stream<ap_uint<_W> >& weightFixedStrm,
                    hls::stream<ap_uint<_W> > indiceStrm[_NStrm],
                    hls::stream<ap_uint<_W> > weightStrm[_NStrm],
                    hls::stream<ap_uint<4> >& channelNum) {
#pragma HLS inline off

    for (int i = 0; i < nnz; ++i) {
#pragma HLS loop_tripcount min = 3000000 avg = 3000000 max = 3000000
#pragma HLS pipeline II = 1
        ap_uint<_W> indice = indiceFixedStrm.read();
        ap_uint<_W> weight = weightFixedStrm.read();
        ap_uint<_W> Channel = indice >> unrollBin;
#if (CHANNEL_NUM == 6)
        ap_uint<3> channelIdx = Channel % 6; // Channel(2, 0);
        indiceStrm[channelIdx].write(indice);
        weightStrm[channelIdx].write(weight);
        channelNum.write(channelIdx);

#else
        ap_uint<1> channelIdx = Channel[0];
        indiceStrm[channelIdx].write(indice);
        weightStrm[channelIdx].write(weight);
        channelNum.write(channelIdx);
#endif
    }

    channelNum.write(0); // one more for redundancy
}

template <typename T>
void transfer(int nnz,
              hls::stream<ap_uint<8 * sizeof(T)> >& cntStrm2,
              hls::stream<ap_uint<8 * sizeof(T)> >& pingStrm2,
              hls::stream<ap_uint<32> >& weightStrm2,
              hls::stream<T>& cntStrm,
              hls::stream<T>& pingStrm,
              hls::stream<float>& weightStrm) {
#pragma HLS inline off
    const int widthT = 8 * sizeof(T);
    int cnt = 0;

    while (cnt < nnz) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1

        calc_degree::f_cast<T> constVal;
        calc_degree::f_cast<T> pingVal;
        calc_degree::f_cast<float> weightVal;

        if (cntStrm2.empty() == false) {
            constVal.i = cntStrm2.read();
            pingVal.i = pingStrm2.read();
            weightVal.i = weightStrm2.read();
            if (sizeof(T) == 8) {
                cntStrm.write(constVal.f);
                pingStrm.write(pingVal.f);
                weightStrm.write(weightVal.f);

            } else {
                cntStrm.write(constVal.f);
                pingStrm.write(pingVal.f);
                weightStrm.write(weightVal.f);
            }
            cnt++;
        }
    }
}

template <typename T>
void calculateMul(
    int nnz, hls::stream<T>& inStrm, hls::stream<T>& cntStrm, hls::stream<float>& weightStrm, hls::stream<T>& mulStrm) {
#pragma HLS inline off

    int cnt = 0;

    while (cnt < nnz) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1

        if (inStrm.empty() == false) {
            T tmp1 = inStrm.read();
            T tmp2 = cntStrm.read();
            T tmp3 = weightStrm.read();
            T mul = tmp1 * tmp2 * tmp3;

            mulStrm.write(mul);
            cnt++;
        }
    }
}

// clang-format off
template <int unrollNm, int widthOr>
void getOffset(int nrows,
                 hls::stream<ap_uint<widthOr> >& offsetStrm,
                 hls::stream<ap_uint<32> >& distStrm2,
                 hls::stream<ap_uint<2> >& flagUnStrm){
// clang-format on
#pragma HLS inline off

    ap_uint<32> NOEDGEVAL = 0xffffffff;
    ap_uint<32> counter1[unrollNm];
#pragma HLS array_partition variable = counter1 dim = 0 complete
    ap_uint<31> prev = 0;
    int iteration = (nrows + unrollNm - 1) / unrollNm;
    ap_uint<32> offsetwithflag;
    ap_uint<31> valP = prev;
    ap_uint<widthOr> tmp;

    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/unrollNm avg = 3700000/unrollNm max = 3700000/unrollNm
        // clang-format on
        for (int k = 0; k < unrollNm; ++k) {
#pragma HLS pipeline II = 1
            int index = i * unrollNm + k;
            if (index < nrows) {
                if (k == 0) {
                    tmp = offsetStrm.read();
                    valP = prev;
                }
                offsetwithflag.range(31, 0) = tmp.range(32 * (k + 1) - 1, 32 * k);

                ap_uint<1> noIOedgeflag = (offsetwithflag == NOEDGEVAL) ? 1 : 0;
                ap_uint<1> onlyIedgeflag = offsetwithflag[31];

                ap_uint<2> flag;
                flag[0] = noIOedgeflag;
                flag[1] = onlyIedgeflag;
                flagUnStrm.write(flag);
                ap_uint<31> offset = offsetwithflag.range(30, 0);

                if (offsetwithflag == 0xffffffff) {
                    counter1[k] = 0;
                } else if ((i == 0) && (k == 0)) {
                    counter1[k] = offset;
                } else if ((i != 0) && (k == 0)) {
                    counter1[k] = offset - prev;
                } else {
                    counter1[k] = offset - valP;
                }

                distStrm2.write(counter1[k]);
                if (offsetwithflag != 0xffffffff) {
                    valP = offset;
                }
            }
            if (k == unrollNm - 1) {
                prev = valP;
            }
        }
    }
}

// clang-format off
template <typename T, int unrollNm, int unrollBin>
void dispatch(	 int nrows,
                 int nnz,
				 hls::stream<ap_uint<2> >& flagUnStrm,
                 hls::stream<ap_uint<32> >& distStrm,
                 hls::stream<T>& mulStrm,

                 hls::stream<T> tmpStrm[unrollNm],
                 hls::stream<ap_uint<3> > distEStrm[unrollNm]){
// clang-format on
#pragma HLS inline off
    ap_uint<unrollBin> k = 0; // index of channel
    int cntRow = 0;           // vertex ID
    int distance = distStrm.read();
    ap_uint<2> flag = flagUnStrm.read();
    ap_uint<3> synFlag = 0; // e + onlyinedge + noedge
    bool enFlag = true;

    while (cntRow < nrows) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
        if ((cntRow == nrows - 1) && (distance == 0)) {
            synFlag[2] = 1;
            synFlag(1, 0) = flag;

            distEStrm[k].write(synFlag);
            cntRow++;
        } else if (distance == 0) {
            synFlag[2] = 1;
            synFlag(1, 0) = flag;

            distEStrm[k].write(synFlag);
            k++;
            distance = distStrm.read();
            flag = flagUnStrm.read();
            cntRow++;
        } else {
            T tmp = mulStrm.read();
            tmpStrm[k].write(tmp);
            synFlag[2] = 0;
            synFlag(1, 0) = flag;
            distEStrm[k].write(synFlag);
            distance--;
        }
    }
}

// clang-format off
template <typename T, int unrollNm>
void adderWrapper(int k,
                  int nrows,
				  bool useSource,
                  int iteration,
                  T adder,
                  hls::stream<ap_uint<3> >& distEStrm,
                  hls::stream<T>& tmpStrm,
                  hls::stream<T>& outStrm) {
// clang-format on
#pragma HLS inline off
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/unrollNm avg = 3700000/unrollNm max = 3700000/unrollNm
        // clang-format on
        int index = i * unrollNm + k;
        if (index < nrows) {
            ap_uint<3> synFlag = distEStrm.read();
            T outSum = (synFlag[1] || synFlag[0]) ? 0 : adder;

            while (!(synFlag[2])) {
#pragma HLS loop_tripcount min = 1 avg = 5 max = 5
#pragma HLS pipeline II = 5
                if (distEStrm.empty() == false) {
                    synFlag = distEStrm.read();

                    if (synFlag[1] && (!useSource)) {
                        tmpStrm.read();
                        outSum = 1.0;
                    } else {
                        T tmp = tmpStrm.read();
                        outSum += tmp;
                    }
                }
            }

            outStrm.write(outSum);
        }
    }
}

// clang-format off
template <typename T, int unrollNm>
void adderPart2(int nrows,
				bool useSource,
                int iteration,
                T adder,
                hls::stream<ap_uint<3> > distEStrm[unrollNm],

                hls::stream<T> tmpStrm[unrollNm],
                hls::stream<T> outStrm[unrollNm]) {
// clang-format on
#pragma HLS inline off
#pragma HLS dataflow

Loop_adder2:
    for (int k = 0; k < unrollNm; ++k) {
#pragma HLS unroll factor = unrollNm
        adderWrapper<T, unrollNm>(k, nrows, useSource, iteration, adder, distEStrm[k], tmpStrm[k],
                                  outStrm[k]); // flagUnStrm2[k],
    }
}

// clang-format off
template <typename T, int rowTemplate, int unrollNm, int unrollBin>
void combineStrm512(int nrows, hls::stream<T> outStrm[unrollNm],
		hls::stream<buffT> buffStrm[CHANNEL_NUM],
		hls::stream<bool> estrm[CHANNEL_NUM]
) {
// clang-format on
#pragma HLS inline off
    ap_uint<unrollBin> cnt = 0;
    const int size = sizeof(T) * 8;
    const int wN = unrollNm;
    const int iteration = (nrows + wN - 1) / wN;
    buffT tmp = 0;
    ap_uint<3> channel = 0;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
        // clang-format on
        for (int k = 0; k < wN; ++k) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
            int index = i * wN + k;
            if (index < nrows) {
                calc_degree::f_cast<T> pagerank;
                pagerank.f = outStrm[cnt].read();
                tmp.range(size * (k + 1) - 1, size * k) = pagerank.i;
                cnt++;
            }
            if (k == wN - 1) {
#if (CHANNEL_NUM == 6)
                ap_uint<3> channelIdx = channel;
#else
                ap_uint<1> channelIdx = channel[0];
#endif
                buffStrm[channelIdx].write(tmp);
                estrm[channelIdx].write(false);
                if (channel == 5) {
                    channel = 0;
                } else {
                    channel++;
                }
            }
        }
    }
    for (int i = 0; i < CHANNEL_NUM; ++i) {
#pragma HLS UNROLL
        estrm[i].write(true);
    }
}

#ifndef __SYNTHESIS__
template <typename T, int rowTemplate>
void writeOut(hls::stream<buffT>& buffStrm, hls::stream<bool>& estrm, buffT* bufferWrite) {
#else
template <typename T, int rowTemplate>
void writeOut(hls::stream<buffT>& buffStrm, hls::stream<bool>& estrm, buffT bufferWrite[rowTemplate]) {
#endif
#pragma HLS inline off
    bool e = estrm.read();
    int i = 0;
    while (!e) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        e = estrm.read();
        buffT tmp = buffStrm.read();
        bufferWrite[i] = tmp;
        i++;
    }
}

template <int BURST_LENTH,
          typename T,
          int rowTemplate,
          int NNZTemplate,
          int unrollNm,
          int unrollBin,
          int widthOr,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,

          int usURAM>
void dataFlowPart(
    int nrows,
    int nnz,
    bool useSource,
    int numEdgePerChannel[CHANNEL_NUM],
    T adder,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT cntVal0[rowTemplate],
    buffT buffPing0[rowTemplate],
    buffT buffPong0[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache1,
    buffT cntVal1[rowTemplate],
    buffT buffPing1[rowTemplate],
    buffT buffPong1[rowTemplate]
#if (CHANNEL_NUM == 6)
    ,
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache2,
    buffT cntVal2[rowTemplate],
    buffT buffPing2[rowTemplate],
    buffT buffPong2[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache3,
    buffT cntVal3[rowTemplate],
    buffT buffPing3[rowTemplate],
    buffT buffPong3[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache4,
    buffT cntVal4[rowTemplate],
    buffT buffPing4[rowTemplate],
    buffT buffPong4[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache5,
    buffT cntVal5[rowTemplate],
    buffT buffPing5[rowTemplate],
    buffT buffPong5[rowTemplate]
#endif
    ) {
#pragma HLS inline off
#pragma HLS dataflow
    const int widthT = sizeof(T) * 8;
    const int iteration = (sizeof(T) == 8) ? (nrows + 7) / 8 : (nrows + 15) / 16; // iteration by line
    // clang-format off
    hls::stream<ap_uint<widthT> >   pingStrm[CHANNEL_NUM];
#pragma HLS resource     variable = pingStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = pingStrm depth = 16

    hls::stream<ap_uint<widthT> >   cntStrm[CHANNEL_NUM];
#pragma HLS resource     variable = cntStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = cntStrm depth = 16

    hls::stream<T>                  pingStrmT[CHANNEL_NUM];
#pragma HLS resource     variable = pingStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = pingStrmT depth = 16

    hls::stream<T>                  cntStrmT[CHANNEL_NUM];
#pragma HLS resource     variable = cntStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = cntStrmT depth = 16

    hls::stream<float>              weightStrmT[CHANNEL_NUM];
#pragma HLS resource     variable = weightStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = weightStrmT depth = 16

    hls::stream<T>                  	   mulStrm[CHANNEL_NUM];
#pragma HLS resource     		variable = mulStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = mulStrm depth = 64

    hls::stream<T>						   mulCollStrm;
#pragma HLS resource     		variable = mulCollStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = mulCollStrm depth = 64

    hls::stream<ap_uint<widthOr> >  	   offsetStrm("offsetStrm");
#pragma HLS resource     		variable = offsetStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = offsetStrm depth = 16

    hls::stream<ap_uint<32> >       	   distStrm;
#pragma HLS resource     		variable = distStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = distStrm depth = 16

    hls::stream<ap_uint<2> >        	   flagUnStrm[unrollNm];
#pragma HLS resource     		variable = flagUnStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = flagUnStrm depth = 16

    hls::stream<T>                  	   tmpStrm[unrollNm];
#pragma HLS resource     		variable = tmpStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = tmpStrm depth = 1024

    hls::stream<ap_uint<3> >        	   distEStrm[unrollNm];
#pragma HLS resource     		variable = distEStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = distEStrm depth = 1024

    hls::stream<ap_uint<2> >        	   flagUnStrm2[unrollNm];
#pragma HLS resource     		variable = flagUnStrm2 core = FIFO_LUTRAM
#pragma HLS stream       		variable = flagUnStrm2 depth = 1024

    hls::stream<T>                  	   outStrm[unrollNm];
#pragma HLS resource     		variable = outStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = outStrm depth = 32

    hls::stream<ap_uint<32> > 	 	 	   indiceNStrm("indiceNStrm");
#pragma HLS stream depth = 16 	variable = indiceNStrm
#pragma HLS resource 		  	variable = indiceNStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > 		 	   indiceStrm[CHANNEL_NUM];
#pragma HLS stream depth = 32  variable = indiceStrm
#pragma HLS resource 		    variable = indiceStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > 	 	 	   weightNStrm("weightNStrm");
#pragma HLS stream depth = 16   variable = weightNStrm
#pragma HLS resource 		    variable = weightNStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > 		 	   weightStrm[CHANNEL_NUM];
#pragma HLS stream depth = 32  variable = weightStrm
#pragma HLS resource 		    variable = weightStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > 			   weightPassStrm[CHANNEL_NUM];
#pragma HLS stream depth = 16 	variable = weightPassStrm
#pragma HLS resource 		 	variable = weightPassStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<4 > > 		 	   channelNum;
#pragma HLS stream depth = 4096 variable = channelNum
#pragma HLS resource 		    variable = channelNum core = FIFO_LUTRAM

    hls::stream<buffT>              	   buffStrm[CHANNEL_NUM];
#pragma HLS resource     		variable = buffStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = buffStrm depth = 16

    hls::stream<bool>               	   estrm[CHANNEL_NUM];
#pragma HLS resource      		variable = estrm core = FIFO_LUTRAM
#pragma HLS stream        		variable = estrm depth = 16

    hls::stream<ap_uint<2> >        	   flagStrm;
#pragma HLS resource     		variable = flagStrm core = FIFO_LUTRAM
#pragma HLS stream       		variable = flagStrm depth = 16

    // clang-format on

    // To pass dataflow check
    int numEdgeChannel0 = numEdgePerChannel[0];
    int numEdgeChannel1 = numEdgePerChannel[1];
#if (CHANNEL_NUM == 6)
    int numEdgeChannel2 = numEdgePerChannel[2];
    int numEdgeChannel3 = numEdgePerChannel[3];
    int numEdgeChannel4 = numEdgePerChannel[4];
    int numEdgeChannel5 = numEdgePerChannel[5];
#endif

    // channelNum is the HBM channel num
    xf::common::utils_hw::axiToStream<BURST_LENTH, _Width, ap_uint<32> >(indices, nnz, indiceNStrm);
    xf::common::utils_hw::axiToStream<BURST_LENTH, _Width, ap_uint<32> >(weight, nnz, weightNStrm);
    // ap_uint<64>, 1, 1, 16, 32, 0, 0, 0

    dispatchIndice<32, unrollBin, CHANNEL_NUM>(nnz, indiceNStrm, weightNStrm, indiceStrm, weightStrm, channelNum);

    cache0.readOnly(numEdgeChannel0, buffPing0, cntVal0, 0, indiceStrm[0], weightStrm[0], pingStrm[0], cntStrm[0],
                    weightPassStrm[0]);
    cache1.readOnly(numEdgeChannel1, buffPing1, cntVal1, 1, indiceStrm[1], weightStrm[1], pingStrm[1], cntStrm[1],
                    weightPassStrm[1]);
#if (CHANNEL_NUM == 6)
    cache2.readOnly(numEdgeChannel2, buffPing2, cntVal2, 2, indiceStrm[2], weightStrm[2], pingStrm[2], cntStrm[2],
                    weightPassStrm[2]);
    cache3.readOnly(numEdgeChannel3, buffPing3, cntVal3, 3, indiceStrm[3], weightStrm[3], pingStrm[3], cntStrm[3],
                    weightPassStrm[3]);
    cache4.readOnly(numEdgeChannel4, buffPing4, cntVal4, 4, indiceStrm[4], weightStrm[4], pingStrm[4], cntStrm[4],
                    weightPassStrm[4]);
    cache5.readOnly(numEdgeChannel5, buffPing5, cntVal5, 5, indiceStrm[5], weightStrm[5], pingStrm[5], cntStrm[5],
                    weightPassStrm[5]);
#endif

    for (int i = 0; i < CHANNEL_NUM; ++i) {
#pragma HLS unroll
        transfer<T>(numEdgePerChannel[i], cntStrm[i], pingStrm[i], weightPassStrm[i], cntStrmT[i], pingStrmT[i],
                    weightStrmT[i]);
        calculateMul<T>(numEdgePerChannel[i], pingStrmT[i], cntStrmT[i], weightStrmT[i], mulStrm[i]);
    }

    // clang-format off
    xf::graph::internal::burstRead2Strm<ap_uint<widthOr> >(iteration, order, offsetStrm);
    // clang-format on

    collect<T, unrollNm, unrollBin>(nnz, mulStrm, channelNum, mulCollStrm);

    getOffset<unrollNm, widthOr>(nrows, offsetStrm, distStrm, flagStrm);
    dispatch<T, unrollNm, unrollBin>(nrows, nnz, flagStrm, distStrm, mulCollStrm, tmpStrm, distEStrm);
    adderPart2<T, unrollNm>(nrows, useSource, iteration, adder, distEStrm, tmpStrm, outStrm);
    combineStrm512<T, rowTemplate, unrollNm, unrollBin>(nrows, outStrm, buffStrm, estrm);

    writeOut<T, rowTemplate>(buffStrm[0], estrm[0], buffPong0);
    writeOut<T, rowTemplate>(buffStrm[1], estrm[1], buffPong1);
#if (CHANNEL_NUM == 6)
    writeOut<T, rowTemplate>(buffStrm[2], estrm[2], buffPong2);
    writeOut<T, rowTemplate>(buffStrm[3], estrm[3], buffPong3);
    writeOut<T, rowTemplate>(buffStrm[4], estrm[4], buffPong4);
    writeOut<T, rowTemplate>(buffStrm[5], estrm[5], buffPong5);
#endif
}

template <int BURST_LENTH,
          typename T,
          int rowTemplate,
          int NNZTemplate,
          int unrollNm,
          int unrollBin,
          int widthOr,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,

          int usURAM>
void dataFlowWrapper(
    int nrows,
    int nnz,
    bool useSource,
    int numEdgePerChannel[CHANNEL_NUM],
    ap_uint<1>& share,
    T adder,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT cntVal0[rowTemplate],
    buffT buffPing0[rowTemplate],
    buffT buffPong0[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache1,
    buffT cntVal1[rowTemplate],
    buffT buffPing1[rowTemplate],
    buffT buffPong1[rowTemplate]
#if (CHANNEL_NUM == 6)
    ,
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache2,
    buffT cntVal2[rowTemplate],
    buffT buffPing2[rowTemplate],
    buffT buffPong2[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache3,
    buffT cntVal3[rowTemplate],
    buffT buffPing3[rowTemplate],
    buffT buffPong3[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache4,
    buffT cntVal4[rowTemplate],
    buffT buffPing4[rowTemplate],
    buffT buffPong4[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache5,
    buffT cntVal5[rowTemplate],
    buffT buffPing5[rowTemplate],
    buffT buffPong5[rowTemplate]
#endif
    ) {
#pragma HLS inline off

    cache0.initDualOffChip();
    cache1.initDualOffChip();
#if (CHANNEL_NUM == 6)
    cache2.initDualOffChip();
    cache3.initDualOffChip();
    cache4.initDualOffChip();
    cache5.initDualOffChip();
#endif

    if (share) {
        dataFlowPart<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                     dataOneLine, addrWidth, usURAM>(
            nrows, nnz, useSource, numEdgePerChannel, adder, indices, weight, order, cache0, cntVal0, buffPing0,
            buffPong0, cache1, cntVal1, buffPing1, buffPong1
#if (CHANNEL_NUM == 6)
            ,
            cache2, cntVal2, buffPing2, buffPong2, cache3, cntVal3, buffPing3, buffPong3, cache4, cntVal4, buffPing4,
            buffPong4, cache5, cntVal5, buffPing5, buffPong5
#endif
            );
    } else {
        dataFlowPart<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                     dataOneLine, addrWidth, usURAM>(
            nrows, nnz, useSource, numEdgePerChannel, adder, indices, weight, order, cache0, cntVal0, buffPong0,
            buffPing0, cache1, cntVal1, buffPong1, buffPing1
#if (CHANNEL_NUM == 6)
            ,
            cache2, cntVal2, buffPong2, buffPing2, cache3, cntVal3, buffPong3, buffPing3, cache4, cntVal4, buffPong4,
            buffPing4, cache5, cntVal5, buffPong5, buffPing5
#endif
            );
    }
    share++;
}

template <int BURST_LENTH,
          typename T,
          int rowTemplate,
          int NNZTemplate,
          int unrollNm,
          int unrollBin,
          int widthOr,
          int uramRow,
          int groupUramPart,
          int dataOneLine,
          int addrWidth,

          int usURAM>
void dataFlowTop(
    int nrows,
    int nnz,
    bool useSource,
    int numEdgePerChannel[CHANNEL_NUM],
    ap_uint<1>& share,
    T adder,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT cntVal0[rowTemplate],
    buffT buffPing0[rowTemplate],
    buffT buffPong0[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache1,
    buffT cntVal1[rowTemplate],
    buffT buffPing1[rowTemplate],
    buffT buffPong1[rowTemplate],
#if (CHANNEL_NUM == 6)
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache2,
    buffT cntVal2[rowTemplate],
    buffT buffPing2[rowTemplate],
    buffT buffPong2[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache3,
    buffT cntVal3[rowTemplate],
    buffT buffPing3[rowTemplate],
    buffT buffPong3[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache4,
    buffT cntVal4[rowTemplate],
    buffT buffPing4[rowTemplate],
    buffT buffPong4[rowTemplate],
    cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache5,
    buffT cntVal5[rowTemplate],
    buffT buffPing5[rowTemplate],
    buffT buffPong5[rowTemplate],
#endif
    T tol,
    bool& converged) {

#pragma HLS inline off
    bool channelConverged0;
    bool channelConverged1;
    bool channelConverged2 = true;
    bool channelConverged3 = true;
    bool channelConverged4 = true;
    bool channelConverged5 = true;

    dataFlowWrapper<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                    dataOneLine, addrWidth, usURAM>(
        nrows, nnz, useSource, numEdgePerChannel, share, adder, indices, weight, order, cache0, cntVal0, buffPing0,
        buffPong0, cache1, cntVal1, buffPing1, buffPong1
#if (CHANNEL_NUM == 6)
        ,
        cache2, cntVal2, buffPing2, buffPong2, cache3, cntVal3, buffPing3, buffPong3, cache4, cntVal4, buffPing4,
        buffPong4, cache5, cntVal5, buffPing5, buffPong5
#endif
        );

    calConvergence<T, rowTemplate>(nrows, tol, channelConverged0, buffPing0, buffPong0, channelConverged1, buffPing1,
                                   buffPong1
#if (CHANNEL_NUM == 6)
                                   ,
                                   channelConverged2, buffPing2, buffPong2, channelConverged3, buffPing3, buffPong3,
                                   channelConverged4, buffPing4, buffPong4, channelConverged5, buffPing5, buffPong5
#endif
                                   );
    converged = channelConverged0 & channelConverged1 & channelConverged2 & channelConverged3 & channelConverged4 &
                channelConverged5;
}

// clang-format off
// for NOEDGE vertex the PR is none
// for ONLYOUTEDGE vertex the PR = 0.15 by the algorithm
// for ONLYINEDGE vertex the PR = 1.0
// for other vertex the PR init = randomProbability which set by users
template <typename T, int widthOr>
void preWrite32(int nrows,
                T alpha,
                T randomProbability,
                hls::stream<buffT>& csrDegree,
                hls::stream<buffT>& cscOffset,
                hls::stream<buffT>& pongStrm,
                hls::stream<buffT>& cntStrm,
                hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int unroll2 = 16;
    ap_uint<32> NOEDGEVAL = 0xffffffff;
    ap_uint<1> onlyIedgeflag = 1;

    const int sizeT = 32;
    const int sizeT2 = 16;
    ap_uint<32> tmpCSCPre = 0;
    int iteration = (nrows + 15) / 16;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/16 avg = 3700000/16 max = 3700000/16
// clang-format on
#pragma HLS pipeline II = 1
        buffT tmpOffset = cscOffset.read();
        buffT tmpDgree = csrDegree.read();
        buffT pongTmp;
        buffT cntTmp;
        buffT pongTmp1;
        buffT cntTmp1;
        ap_uint<widthOr> orderTmp;
        ap_uint<sizeT> pongT[unroll2];
        ap_uint<sizeT> cntT[unroll2];
        ap_uint<32> orderT[unroll2];
#pragma HLS array_partition variable = pongT dim = 0 complete
#pragma HLS array_partition variable = cntT dim = 0 complete
#pragma HLS array_partition variable = orderT dim = 0 complete
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS loop_tripcount min = unroll2 avg = unroll2 max = unroll2
#pragma HLS unroll factor = unroll2
            int index = i * unroll2 + k;
            if (index < nrows) {
                ap_uint<32> cntCSC;
                calc_degree::f_cast<T> cntCSR;
                cntCSR.i = tmpDgree.range(32 * (k + 1) - 1, 32 * k);

                ap_uint<32> tmpCSC = tmpOffset.range(32 * (k + 1) - 1, 32 * k);
                if (k == 0) {
                    cntCSC = tmpCSC - tmpCSCPre;
                } else {
                    cntCSC = tmpCSC - tmpOffset.range(32 * k - 1, 32 * (k - 1));
                }
                if (k == unroll2 - 1) {
                    tmpCSCPre = tmpCSC;
                }
                if ((cntCSR.f == 0.0) && (cntCSC == 0)) {
                    pongT[k] = 0;
                    orderT[k] = NOEDGEVAL;
                } else if ((cntCSR.f == 0.0) && (cntCSC != 0)) {
                    pongT[k] = 1.0;
                    // orderT[k] = ONLYINEDGEVAL;
                    orderT[k][31] = onlyIedgeflag;
                    orderT[k].range(30, 0) = tmpCSC.range(30, 0); //(ap_uint<31>)tmpCSC;
                } else {
                    calc_degree::f_cast<T> tTmp;
                    tTmp.f = randomProbability;
                    pongT[k] = tTmp.i;
                    orderT[k] = tmpCSC;
                }

                if (cntCSR.f != 0.0) {
                    calc_degree::f_cast<T> tTmp2;
                    tTmp2.f = alpha / cntCSR.f;
                    cntT[k] = tTmp2.i;
                } else {
                    cntT[k] = 0;
                }
            }
        }
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS unroll factor = unroll2
            orderTmp.range(32 * (k + 1) - 1, 32 * k) = orderT[k].range(31, 0);
            pongTmp.range(sizeT * (k + 1) - 1, sizeT * k) = pongT[k].range(sizeT - 1, 0);
            cntTmp.range(sizeT * (k + 1) - 1, sizeT * k) = cntT[k].range(sizeT - 1, 0);
        }
        pongStrm.write(pongTmp);
        cntStrm.write(cntTmp);
        orderStrm.write(orderTmp);
    }
}

// the degree is int and use ap_ufixed
// clang-format off
template <typename T, int widthOr>
void preWrite64(int nrows,
                T alpha,
                T randomProbability,
                hls::stream<buffT>& csrDegree,
                hls::stream<buffT>& cscOffset,
                hls::stream<buffT>& pongStrm,
                hls::stream<buffT>& cntStrm,
                hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int unroll2 = 8;
    ap_uint<32> NOEDGEVAL = 0xffffffff;

    const int sizeT = 64;
    ap_uint<32> tmpCSCPre = 0;
    ap_uint<1> onlyIedgeflag = 1;
    int iteration = (nrows + 7) / 8;
    ap_uint<1> cnt = 0;
    int offs = 256;
    buffT tmpOffset;
    buffT tmpDgree;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1

        if (cnt == 0) {
            tmpOffset = cscOffset.read();
            tmpDgree = csrDegree.read();
        }
        buffT pongTmp;
        buffT cntTmp;
        buffT pongTmp1;
        buffT cntTmp1;
        ap_uint<widthOr> orderTmp;
        ap_uint<sizeT> pongT[unroll2];
        ap_uint<sizeT> cntT[unroll2];
        ap_uint<32> orderT[unroll2];
#pragma HLS array_partition variable = pongT dim = 0 complete
#pragma HLS array_partition variable = cntT dim = 0 complete
#pragma HLS array_partition variable = orderT dim = 0 complete
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS loop_tripcount min = unroll2 avg = unroll2 max = unroll2
#pragma HLS unroll factor = unroll2
            int index = i * unroll2 + k;
            if (index < nrows) {
                ap_uint<32> cntCSC;

                if (cnt == 0) {
                    offs = 0;
                } else {
                    offs = 256;
                }
                calc_degree::f_cast<float> cntCSR;
                cntCSR.i = tmpDgree.range(32 * (k + 1) - 1 + offs, 32 * k + offs);

                ap_uint<32> tmpCSC = tmpOffset.range(32 * (k + 1) - 1 + offs, 32 * k + offs);
                if (k == 0) {
                    cntCSC = tmpCSC - tmpCSCPre;
                } else {
                    cntCSC = tmpCSC - tmpOffset.range(32 * k - 1, 32 * (k - 1));
                }
                if (k == unroll2 - 1) {
                    tmpCSCPre = tmpCSC;
                }
                if ((cntCSR.f == 0.0) && (cntCSC == 0)) {
                    pongT[k] = 0;
                    orderT[k] = NOEDGEVAL;
                } else if ((cntCSR.f == 0.0) && (cntCSC != 0)) {
                    pongT[k] = 1.0;
                    orderT[k][31] = onlyIedgeflag;
                    orderT[k].range(30, 0) = tmpCSC.range(30, 0); //(ap_uint<31>)tmpCSC;
                } else {
                    calc_degree::f_cast<T> tTmp;
                    tTmp.f = randomProbability;
                    pongT[k] = tTmp.i;
                    orderT[k] = tmpCSC;
                }

                if (cntCSR.f != 0.0) {
                    calc_degree::f_cast<T> tTmp2;
                    tTmp2.f = alpha / cntCSR.f;
                    cntT[k] = tTmp2.i;
                } else {
                    cntT[k] = 0;
                }
            }
        }
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS unroll factor = unroll2
            orderTmp.range(32 * (k + 1) - 1, 32 * k) = orderT[k].range(31, 0);
            pongTmp.range(sizeT * (k + 1) - 1, sizeT * k) = pongT[k].range(sizeT - 1, 0);
            cntTmp.range(sizeT * (k + 1) - 1, sizeT * k) = cntT[k].range(sizeT - 1, 0);
        }
        cnt++;
        pongStrm.write(pongTmp);
        cntStrm.write(cntTmp);
        orderStrm.write(orderTmp);
    }
}

// clang-format off
template <typename T, int widthOr>
void preWrite32WithSource(int nrows,
                T alpha,
                T randomProbability,
				const int sourceLength,
				hls::stream<ap_uint<32> >& sourceStrm,
                hls::stream<buffT>& csrDegree,
                hls::stream<buffT>& cscOffset,
                hls::stream<buffT>& pongStrm,
                hls::stream<buffT>& cntStrm,
                hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int unroll2 = 16;
    ap_uint<32> tVal = 0xffffffff;
    ap_uint<1> onlyIedgeflag = 1;
    ap_uint<1> notSourceflag = 1;

    const int sizeT = 32;
    const int sizeT2 = 16;
    ap_uint<32> tmpCSCPre = 0;
    int iteration = (nrows + 15) / 16;

    ap_uint<32> sourceID = sourceStrm.read();
    int cntSource = 1;

    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/16 avg = 3700000/16 max = 3700000/16
        // clang-format on

        buffT tmpOffset = cscOffset.read();
        buffT tmpDgree = csrDegree.read();

        buffT pongTmp;
        buffT cntTmp;
        buffT pongTmp1;
        buffT cntTmp1;
        ap_uint<widthOr> orderTmp;
        ap_uint<sizeT> pongT[unroll2];
        ap_uint<sizeT> cntT[unroll2];
        ap_uint<32> orderT[unroll2];
#pragma HLS array_partition variable = pongT dim = 0 complete
#pragma HLS array_partition variable = cntT dim = 0 complete
#pragma HLS array_partition variable = orderT dim = 0 complete
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS loop_tripcount min = unroll2 avg = unroll2 max = unroll2
#pragma HLS pipeline II = 1
            int index = i * unroll2 + k;
            if (index < nrows) {
                ap_uint<32> cntCSC;
                calc_degree::f_cast<T> cntCSR;
                cntCSR.i = tmpDgree.range(32 * (k + 1) - 1, 32 * k);

                ap_uint<32> tmpCSC = tmpOffset.range(32 * (k + 1) - 1, 32 * k);
                if (k == 0) {
                    cntCSC = tmpCSC - tmpCSCPre;
                } else {
                    cntCSC = tmpCSC - tmpOffset.range(32 * k - 1, 32 * (k - 1));
                }
                if (k == unroll2 - 1) {
                    tmpCSCPre = tmpCSC;
                }

                if (index == sourceID) {
                    calc_degree::f_cast<T> tTmp;
                    tTmp.f = randomProbability;
                    pongT[k] = tTmp.i;
                    orderT[k] = tmpCSC;
                    if (cntSource < sourceLength) sourceID = sourceStrm.read();
                } else {
                    pongT[k] = 0;
                    orderT[k][31] = notSourceflag;
                    orderT[k].range(sizeT - 2, 0) = (ap_uint<31>)tmpCSC;
                }

                if (cntCSR.f != 0.0) {
                    calc_degree::f_cast<T> tTmp2;
                    tTmp2.f = alpha / cntCSR.f;
                    cntT[k] = tTmp2.i;
                } else {
                    cntT[k] = 0;
                }
            }
        }
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS unroll factor = unroll2
            orderTmp.range(32 * (k + 1) - 1, 32 * k) = orderT[k].range(31, 0);
            pongTmp.range(sizeT * (k + 1) - 1, sizeT * k) = pongT[k].range(sizeT - 1, 0);
            cntTmp.range(sizeT * (k + 1) - 1, sizeT * k) = cntT[k].range(sizeT - 1, 0);
        }
        pongStrm.write(pongTmp);
        cntStrm.write(cntTmp);
        orderStrm.write(orderTmp);
    }
}

// the degree is int and use ap_ufixed
// clang-format off
template <typename T, int widthOr>
void preWrite64WithSource(int nrows,
                T alpha,
                T randomProbability,
				const int sourceLength,
				hls::stream<ap_uint<32> >& sourceStrm,
                hls::stream<buffT>& csrDegree,
                hls::stream<buffT>& cscOffset,
                hls::stream<buffT>& pongStrm,
                hls::stream<buffT>& cntStrm,
                hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int unroll2 = 8;
    ap_uint<32> tVal = 0xffffffff;
    ap_uint<1> onlyIedgeflag = 1;
    ap_uint<1> notSourceflag = 1;

    const int sizeT = 64;
    const int sizeT2 = 32;
    ap_uint<32> tmpCSCPre = 0;
    int iteration = (nrows + 7) / 8;
    ap_uint<1> cnt = 0;
    int offs = 256;
    buffT tmpOffset;
    buffT tmpDgree;

    ap_uint<32> sourceID = sourceStrm.read();
    int cntSource = 1;

    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
        // clang-format on

        if (cnt == 0) {
            tmpOffset = cscOffset.read();
            tmpDgree = csrDegree.read();
        }
        buffT pongTmp;
        buffT cntTmp;
        buffT pongTmp1;
        buffT cntTmp1;
        ap_uint<widthOr> orderTmp;
        ap_uint<sizeT> pongT[unroll2];
        ap_uint<sizeT> cntT[unroll2];
        ap_uint<32> orderT[unroll2];
#pragma HLS array_partition variable = pongT dim = 0 complete
#pragma HLS array_partition variable = cntT dim = 0 complete
#pragma HLS array_partition variable = orderT dim = 0 complete
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS loop_tripcount min = unroll2 avg = unroll2 max = unroll2
#pragma HLS pipeline II = 1
            int index = i * unroll2 + k;
            if (index < nrows) {
                ap_uint<32> cntCSC;
                ap_uint<32> tmpCSC;
                if (cnt == 0) {
                    offs = 0;
                } else {
                    offs = 256;
                }
                calc_degree::f_cast<float> cntCSR;
                cntCSR.i = tmpDgree.range(32 * (k + 1) - 1 + offs, 32 * k + offs);
                tmpCSC = tmpOffset.range(32 * (k + 1) - 1 + offs, 32 * k + offs);
                if ((k == 0) && (cnt == 0)) {
                    cntCSC = tmpCSC - tmpCSCPre;
                } else {
                    cntCSC = tmpCSC - tmpOffset.range(32 * k - 1 + offs, 32 * (k - 1) + offs);
                }
                if ((k == unroll2 - 1) && (cnt == 1)) {
                    tmpCSCPre = tmpCSC;
                }
                if (index == sourceID) {
                    calc_degree::f_cast<T> tTmp;
                    tTmp.f = randomProbability;
                    pongT[k] = tTmp.i;
                    orderT[k] = tmpCSC;
                    if (cntSource < sourceLength) sourceID = sourceStrm.read();
                } else {
                    pongT[k] = 0;
                    orderT[k][31] = notSourceflag;
                    orderT[k].range(sizeT2 - 2, 0) = (ap_uint<31>)tmpCSC;
                }

                if (cntCSR.f != 0.0) {
                    calc_degree::f_cast<T> tTmp2;
                    tTmp2.f = alpha / cntCSR.f;
                    cntT[k] = tTmp2.i;
                } else {
                    cntT[k] = 0;
                }
            }
        }
        for (int k = 0; k < unroll2; ++k) {
#pragma HLS unroll factor = unroll2
            orderTmp.range(32 * (k + 1) - 1, 32 * k) = orderT[k].range(31, 0);
            pongTmp.range(sizeT * (k + 1) - 1, sizeT * k) = pongT[k].range(sizeT - 1, 0);
            cntTmp.range(sizeT * (k + 1) - 1, sizeT * k) = cntT[k].range(sizeT - 1, 0);
        }
        cnt++;
        pongStrm.write(pongTmp);
        cntStrm.write(cntTmp);
        orderStrm.write(orderTmp);
    }
}

// clang-format off
template <typename T, int widthOr>
void preWrite(int nrows,
              T alpha,
              T randomProbability,
			  int sourceLength,
			  hls::stream<ap_uint<32> >& sourceStrm,
              hls::stream<buffT>& csrDegree,
              hls::stream<buffT>& cscOffset,
              hls::stream<buffT>& pongStrm,
              hls::stream<buffT>& cntStrm,
              hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int size0 = sizeof(T);
    bool useSource = (sourceLength != 0);
    if (size0 == 4) {
        if (useSource) {
            preWrite32WithSource<T>(nrows, alpha, randomProbability, sourceLength, sourceStrm, csrDegree, cscOffset,
                                    pongStrm, cntStrm, orderStrm);
        } else {
            preWrite32<T>(nrows, alpha, randomProbability, csrDegree, cscOffset, pongStrm, cntStrm, orderStrm);
        }
    } else if (size0 == 8) {
        if (useSource) {
            preWrite64WithSource<T>(nrows, alpha, randomProbability, sourceLength, sourceStrm, csrDegree, cscOffset,
                                    pongStrm, cntStrm, orderStrm);
        } else {
            preWrite64<T>(nrows, alpha, randomProbability, csrDegree, cscOffset, pongStrm, cntStrm, orderStrm);
        }
    }
}

// clang-format off
template <typename T, int rowTemplate, int widthOr>
void writeOutDDROrder(int nrows, hls::stream<ap_uint<widthOr> >& orderStrm, ap_uint<widthOr> orderUnroll[rowTemplate]) {
// clang-format off

#pragma HLS inline off
    const int iteration = (sizeof(T) == 8) ? (nrows + 7) / 8 : (nrows + 15) / 16;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        orderUnroll[i] = orderStrm.read();
    }
}

template <typename T, int rowTemplate>
void writeOutDDR(hls::stream<buffT>& cntStrm,
                 hls::stream<buffT>& pongStrm,
                 hls::stream<bool>& estrm,
                 buffT cntValFull[rowTemplate],
                 buffT buffPong[rowTemplate]) {
#pragma HLS inline off
    bool e = estrm.read();
    int i = 0;
    while (!e) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        e = estrm.read();
        buffPong[i] = pongStrm.read();
        cntValFull[i] = cntStrm.read();
        i++;
    }
}

template <typename T>
void streamSplit(int nrows,
                 hls::stream<buffT>& cntStrm,
                 hls::stream<buffT>& initStrm,
                 hls::stream<buffT> constStrm[CHANNEL_NUM],
                 hls::stream<buffT> pongStrm[CHANNEL_NUM],
                 hls::stream<bool> estrm[CHANNEL_NUM]) {
#pragma HLS inline off
    const int iteration = (sizeof(T) == 8) ? (nrows + 7) / 8 : (nrows + 15) / 16;
    for (ap_uint<32> i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1

#if (CHANNEL_NUM == 6)
        ap_uint<3> channelIdx = i % 6; // i.range(2, 0);
#else
        ap_uint<1> channelIdx = i.range(0, 0);
#endif
        pongStrm[channelIdx].write(initStrm.read());
        constStrm[channelIdx].write(cntStrm.read());
        estrm[channelIdx].write(false);
    }

    for (int i = 0; i < CHANNEL_NUM; ++i) {
#pragma HLS UNROll
        estrm[i].write(true);
    }
}

template <typename T, int BURST_LENTH, int rowTemplate, int NNZTemplate, int widthOr>
void initDDR(int nrows,
             int nnz,
             T alpha,
             T randomProbability,
             int sourceLength,
             ap_uint<32> sourceID[rowTemplate],
             buffT degreeCSR[rowTemplate],
             buffT offsetCSC[rowTemplate],
             buffT cntValFull0[rowTemplate],
             buffT buffPong0[rowTemplate],
             buffT cntValFull1[rowTemplate],
             buffT buffPong1[rowTemplate],
#if (CHANNEL_NUM == 6)
             buffT cntValFull2[rowTemplate],
             buffT buffPong2[rowTemplate],
             buffT cntValFull3[rowTemplate],
             buffT buffPong3[rowTemplate],
             buffT cntValFull4[rowTemplate],
             buffT buffPong4[rowTemplate],
             buffT cntValFull5[rowTemplate],
             buffT buffPong5[rowTemplate],
#endif
             ap_uint<widthOr> orderUnroll[rowTemplate]) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<buffT> csrDegree("csrDegree");
#pragma HLS stream depth = 32 variable = csrDegree
    hls::stream<buffT> cscOffset("cscOffset");
#pragma HLS stream depth = 32 variable = cscOffset
    hls::stream<buffT> initStrm("initStrm");
#pragma HLS stream depth = 32 variable = initStrm
    hls::stream<buffT> cntStrm("cntStrm");
#pragma HLS stream depth = 32 variable = cntStrm
    hls::stream<buffT> pongStrm[CHANNEL_NUM];
#pragma HLS stream depth = 32 variable = pongStrm
    hls::stream<buffT> constStrm[CHANNEL_NUM];
#pragma HLS stream depth = 32 variable = constStrm
    hls::stream<bool> estrm[CHANNEL_NUM];
#pragma HLS stream depth = 32 variable = estrm
    hls::stream<ap_uint<widthOr> > orderStrm;
#pragma HLS stream depth = 32 variable = orderStrm
    hls::stream<ap_uint<32> > sourceStrm("sourceStrm");
#pragma HLS stream depth = 32 variable = sourceStrm

#pragma HLS resource variable = sourceStrm core = FIFO_LUTRAM
#pragma HLS resource variable = csrDegree core = FIFO_LUTRAM
#pragma HLS resource variable = cscOffset core = FIFO_LUTRAM
#pragma HLS resource variable = initStrm core = FIFO_LUTRAM
#pragma HLS resource variable = cntStrm core = FIFO_LUTRAM
#pragma HLS resource variable = pongStrm core = FIFO_LUTRAM
#pragma HLS resource variable = constStrm core = FIFO_LUTRAM
#pragma HLS resource variable = estrm core = FIFO_LUTRAM
#pragma HLS resource variable = orderStrm core = FIFO_LUTRAM
    const int wN = 16;
    const int iteration = (nrows + wN - 1) / wN;
    const int extra = iteration * wN - nrows;
    bool useSource = (sourceLength != 0);
    xf::graph::internal::axiToCharStream<BURST_LENTH, _Width, buffT>(degreeCSR, csrDegree, 4 * (nrows + extra));
    xf::graph::internal::axiToCharStream<BURST_LENTH, _Width, buffT>(offsetCSC, cscOffset, 4 * (nrows + extra), 4);
    if (useSource) {
        xf::graph::internal::axiToCharStream<BURST_LENTH, 32, ap_uint<32> >(sourceID, sourceStrm, 4 * (sourceLength));
    }

    preWrite<T, widthOr>(nrows, alpha, randomProbability, sourceLength, sourceStrm, csrDegree, cscOffset, initStrm,
                         cntStrm, orderStrm);
    streamSplit<T>(nrows, cntStrm, initStrm, constStrm, pongStrm, estrm);
    writeOutDDR<T, rowTemplate>(constStrm[0], pongStrm[0], estrm[0], cntValFull0, buffPong0);
    writeOutDDR<T, rowTemplate>(constStrm[1], pongStrm[1], estrm[1], cntValFull1, buffPong1);
#if (CHANNEL_NUM == 6)
    writeOutDDR<T, rowTemplate>(constStrm[2], pongStrm[2], estrm[2], cntValFull2, buffPong2);
    writeOutDDR<T, rowTemplate>(constStrm[3], pongStrm[3], estrm[3], cntValFull3, buffPong3);
    writeOutDDR<T, rowTemplate>(constStrm[4], pongStrm[4], estrm[4], cntValFull4, buffPong4);
    writeOutDDR<T, rowTemplate>(constStrm[5], pongStrm[5], estrm[5], cntValFull5, buffPong5);
#endif
    writeOutDDROrder<T, rowTemplate, widthOr>(nrows, orderStrm, orderUnroll);
}
} // namespace pagerank
} // namespace internal

template <typename T,
          int rowTemplate,
          int NNZTemplate,
          int unrollBin,
          int widthOr,
          int uramRowBin,
          int dataOneLineBin,

          int usURAM>
void pageRankCore(int nrows,
                  int nnz,
                  int sourceLength,
                  int numEdgePerChannel[CHANNEL_NUM],
                  ap_uint<widthOr>* order,
                  buffT* indices,
                  buffT* weight,
                  buffT* buffPing0,
                  buffT* buffPong0,
                  buffT* buffPing1,
                  buffT* buffPong1,
                  buffT* cntVal0,
                  buffT* cntVal1,
#if (CHANNEL_NUM == 6)
                  buffT* buffPing2,
                  buffT* buffPong2,
                  buffT* buffPing3,
                  buffT* buffPong3,
                  buffT* buffPing4,
                  buffT* buffPong4,
                  buffT* buffPing5,
                  buffT* buffPong5,
                  buffT* cntVal2,
                  buffT* cntVal3,
                  buffT* cntVal4,
                  buffT* cntVal5,
#endif
                  int* resultInfo,
                  T alpha = 0.85,
                  T tolerance = 1e-4,
                  int maxIter = 200) {
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << dataOneLineBin;  // double 8 : float 16
    const int uramRow = (1 << uramRowBin);        // 4096
    const int groupUramPart = 1 << dataUramNmBin; // 8 = 2^3
    const int addrWidth = 32;
    const int unrollFactor = 1;

    const int BURST_LENTH = 32;
    const int TYPE_LENGTH = sizeof(T) * 8;
    int maxIt;
    T tol;
    int returnVal = 0;
    const int unrollNm = 1 << unrollBin;

    if (maxIter > 0)
        maxIt = maxIter;
    else
        maxIt = 500;

    if (tolerance == 0.0f)
        tol = 1.0E-6f;
    else if (tolerance < 1.0f && tolerance > 0.0f)
        tol = tolerance;
    else
        returnVal = 1;

    if (alpha <= 0.0f || alpha >= 1.0f) returnVal = -1;

    ap_uint<1> share = 0;

    T adder = 1 - alpha;

    bool converged = false;
    bool useSource = (sourceLength != 0);

    int iterator = 0;

    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache0;

    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache1;
#if (CHANNEL_NUM == 6)
    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache2;
    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache3;
    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache4;
    internal::pagerankMultiChannel::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth,
                                          usURAM, usURAM, usURAM>
        cache5;
#endif
    while (!converged && iterator < maxIt) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16

        iterator++;
        converged = 1;

        internal::pagerankMultiChannel::dataFlowTop<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin,
                                                    widthOr, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM>(
            nrows, nnz, useSource, numEdgePerChannel, share, adder, indices, weight, order, cache0, cntVal0, buffPing0,
            buffPong0, cache1, cntVal1, buffPing1, buffPong1,
#if (CHANNEL_NUM == 6)
            cache2, cntVal2, buffPing2, buffPong2, cache3, cntVal3, buffPing3, buffPong3, cache4, cntVal4, buffPing4,
            buffPong4, cache5, cntVal5, buffPing5, buffPong5,
#endif
            tol, converged);
    }

    T divider;
    T sum[unrollNm] = {0.0};
#pragma HLS array_partition variable = sum dim = 0 complete

    *resultInfo = !share;
    *(resultInfo + 1) = iterator;
#ifndef __SYNTHESIS__
    std::cout << "Input validity check, 0 is passed : " << returnVal << std::endl;
    std::cout << "iterator = " << iterator << std::endl;
    std::cout << "isResultinPong = " << !share << std::endl;
#endif
}

/**
 * @brief pagerank algorithm is implemented
 * support: 1. HBM based board
 * 2. double / float for PR value calculate
 * 3. weighted / unweighted graph / personalized graph
 * 4. 2 channel / 6 channel, 2 channel will take 14 persudo channels of HBM
 * 			while 6 channel will take 26 persudo channels of HBM
 *
 * @tparam T date type of pagerank, double or float
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2UNROLL log2 of unroll number for float adder
 * @tparam WIDTHOR order array bandwidth, it's 256 in our case
 * @tparam LOG2CACHEDEPTH log2(cache depth), the onchip memory for cache
 * is 512bits x CACHEDEPTH (512 bits x 2^LOG2CACHEDEPTH)
 * @tparam LOG2DATAPERCACHELINECORE param for module pageRankCore, log2 of
 * number of data in one 512bit (64 byte), for double,
 * it's log2(64/sizeof(double)) = 3,
 * for float, it's log2(64/sizeof(float)) = 4
 * @tparam LOG2DATAPERCACHELINEDEGREE param for module calduDegree,
 * log2 of number of data in one 512bit (64 byte),
 * for double, it's log2(64/sizeof(double)) = 3,
 * for float, it's log2(64/sizeof(float)) = 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM,
 * 0 : LUTRAM, 1 : URAM, 2: BRAM
 * @tparam CHANNEL_NUM pingpong channel number for the design
 * @param numVertex CSR/CSC data offsets number
 * @param numEdge CSR/CSC data indices number
 * @param nsource source vertex ID number, 0 means not apply pagerank_personalized
 * @param sourceID source vertex ID array for pagerank_personalized
 * @param degreeCSR temporary internal degree value
 * @param offsetCSC CSR/CSC data offset array
 * @param indexCSC CSR/CSC data indice array
 * @param weightCSC CSR/CSC data weight array, support type float
 * @param cntValFull0 temporary internal initialized mulplier values, length equals to numVertex
 * @param buffPing0 ping array to keep temporary pagerank value
 * @param buffPong0 pong array to keep temporary pagerank value
 * @param cntValFull1 temporary internal initialized mulplier values, length equals to numVertex
 * @param buffPing1 ping array to keep temporary pagerank value
 * @param buffPong1 pong array to keep temporary pagerank value
 * @param orderUnroll temporary internal order array to keep initialized offset values
 * @param resultInfo The output information. resultInfo[0] is isResultinPong, resultInfo[1] is iterations.
 * @param randomProbability initial PR value, normally 1.0 or 1.0/numVertex
 * @param alpha damping factor, normally 0.85
 * @param tolerance converge tolerance
 * @param numIter max iteration
 */
#if (CHANNEL_NUM == 6)
template <typename T,
          int MAXVERTEX,
          int MAXEDGE,
          int LOG2UNROLL,
          int WIDTHOR,
          int LOG2CACHEDEPTH,
          int LOG2DATAPERCACHELINECORE,
          int LOG2DATAPERCACHELINEDEGREE,
          int RAMTYPE>
void pageRankTop(int numVertex,
                 int numEdge,
                 int nsource,
                 ap_uint<32>* sourceID,
                 ap_uint<512>* degreeCSR,
                 ap_uint<512>* offsetCSC,
                 ap_uint<512>* indexCSC,
                 ap_uint<512>* weightCSC,
                 ap_uint<512>* cntValFull0,
                 ap_uint<512>* buffPing0,
                 ap_uint<512>* buffPong0,
                 ap_uint<512>* cntValFull1,
                 ap_uint<512>* buffPing1,
                 ap_uint<512>* buffPong1,
                 ap_uint<512>* cntValFull2,
                 ap_uint<512>* buffPing2,
                 ap_uint<512>* buffPong2,
                 ap_uint<512>* cntValFull3,
                 ap_uint<512>* buffPing3,
                 ap_uint<512>* buffPong3,
                 ap_uint<512>* cntValFull4,
                 ap_uint<512>* buffPing4,
                 ap_uint<512>* buffPong4,
                 ap_uint<512>* cntValFull5,
                 ap_uint<512>* buffPing5,
                 ap_uint<512>* buffPong5,
                 ap_uint<WIDTHOR>* orderUnroll,
                 int* resultInfo,
                 T randomProbability = 1.0,
                 T alpha = 0.85,
                 T tolerance = 1e-4,
                 int numIter = 200) {
    const int BURST_LENTH = 32;
    int numEdgePerChannel[CHANNEL_NUM];

    xf::graph::calcuWeightedDegree<MAXVERTEX, MAXEDGE, 15, LOG2DATAPERCACHELINEDEGREE, 1, LOG2UNROLL, CHANNEL_NUM>(
        numVertex, numEdge, numEdgePerChannel, indexCSC, weightCSC, degreeCSR);
#ifndef __SYNTHESIS__
    for (int n = 0; n < CHANNEL_NUM; n++) {
        std::cout << "numEdgePerChannel : " << numEdgePerChannel[n] << std::endl;
    }
#endif
    internal::pagerankMultiChannel::initDDR<T, BURST_LENTH, MAXVERTEX, MAXEDGE, WIDTHOR>(
        numVertex, numEdge, alpha, randomProbability, nsource, sourceID, degreeCSR, offsetCSC, cntValFull0, buffPong0,
        cntValFull1, buffPong1, cntValFull2, buffPong2, cntValFull3, buffPong3, cntValFull4, buffPong4, cntValFull5,
        buffPong5, orderUnroll);

    pageRankCore<T, MAXVERTEX, MAXEDGE, LOG2UNROLL, WIDTHOR, LOG2CACHEDEPTH, LOG2DATAPERCACHELINECORE, RAMTYPE>(
        numVertex, numEdge, nsource, numEdgePerChannel, orderUnroll, indexCSC, weightCSC, buffPing0, buffPong0,
        buffPing1, buffPong1, cntValFull0, cntValFull1, buffPing2, buffPong2, buffPing3, buffPong3, buffPing4,
        buffPong4, buffPing5, buffPong5, cntValFull2, cntValFull3, cntValFull4, cntValFull5, resultInfo, alpha,
        tolerance, numIter);
}

#else

template <typename T,
          int MAXVERTEX,
          int MAXEDGE,
          int LOG2UNROLL,
          int WIDTHOR,
          int LOG2CACHEDEPTH,
          int LOG2DATAPERCACHELINECORE,
          int LOG2DATAPERCACHELINEDEGREE,
          int RAMTYPE>
void pageRankTop(int numVertex,
                 int numEdge,
                 int nsource,
                 ap_uint<32>* sourceID,
                 ap_uint<512>* degreeCSR,
                 ap_uint<512>* offsetCSC,
                 ap_uint<512>* indexCSC,
                 ap_uint<512>* weightCSC,
                 ap_uint<512>* cntValFull0,
                 ap_uint<512>* buffPing0,
                 ap_uint<512>* buffPong0,
                 ap_uint<512>* cntValFull1,
                 ap_uint<512>* buffPing1,
                 ap_uint<512>* buffPong1,
                 ap_uint<WIDTHOR>* orderUnroll,
                 int* resultInfo,
                 T randomProbability = 1.0,
                 T alpha = 0.85,
                 T tolerance = 1e-4,
                 int numIter = 200) {
    const int BURST_LENTH = 32;
    int numEdgePerChannel[CHANNEL_NUM];

    xf::graph::calcuWeightedDegree<MAXVERTEX, MAXEDGE, 15, LOG2DATAPERCACHELINEDEGREE, 1, LOG2UNROLL, CHANNEL_NUM>(
        numVertex, numEdge, numEdgePerChannel, indexCSC, weightCSC, degreeCSR);
    internal::pagerankMultiChannel::initDDR<T, BURST_LENTH, MAXVERTEX, MAXEDGE, WIDTHOR>(
        numVertex, numEdge, alpha, randomProbability, nsource, sourceID, degreeCSR, offsetCSC, cntValFull0, buffPong0,
        cntValFull1, buffPong1, orderUnroll);
    pageRankCore<T, MAXVERTEX, MAXEDGE, LOG2UNROLL, WIDTHOR, LOG2CACHEDEPTH, LOG2DATAPERCACHELINECORE, RAMTYPE>(
        numVertex, numEdge, nsource, numEdgePerChannel, orderUnroll, indexCSC, weightCSC, buffPing0, buffPong0,
        buffPing1, buffPong1, cntValFull0, cntValFull1, resultInfo, alpha, tolerance, numIter);
}
#endif
} // namespace graph
} // namespace xf
#endif //#ifndef VT_GRAPH_PR_H
