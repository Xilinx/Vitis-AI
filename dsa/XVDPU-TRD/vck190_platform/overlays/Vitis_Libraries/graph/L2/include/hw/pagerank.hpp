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

#ifndef XF_GRAPH_PR_H
#define XF_GRAPH_PR_H

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "hls_math.h"
#include <hls_stream.h>

#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/cache.hpp"
#include "calc_degree.hpp"

#define _Width 512
typedef ap_uint<_Width> buffT;

namespace xf {
namespace graph {
namespace internal {
namespace pagerank {
// clang-format off
template <typename T, int rowTemplate, int UN>
void vecSub(int nrows, hls::stream<buffT>& pingStrm, hls::stream<buffT>& pongStrm, hls::stream<T> subStream[UN]) {
// clang-format on
#pragma HLS inline off
    const int size = sizeof(T) * 8;
    const int size2 = sizeof(T) * 4;
    const int wN = UN;
    const int iteration = (nrows + wN - 1) / wN;
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
                subStream[k].write(tmpVal);
            }
        }
    }
}

template <typename T, int UN>
void convergenceCal(int nrows, T tol, bool& converged, hls::stream<T> subStream[UN]) {
#pragma HLS inline off
    const int wN = UN;
    const int iteration = (nrows + wN - 1) / wN;
    bool converg[UN] = {1};
    for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
        converg[k] = 1;
    }
#pragma HLS array_partition variable = converg dim = 0 complete
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS unroll factor = wN
            int index = i * wN + k;
            if (index < nrows) {
                T val = subStream[k].read();
                if (val > tol) {
                    converg[k] = 0;
                }
            }
        }
    }
    for (int k = 0; k < wN; ++k) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
        if (converg[k] == 0) {
            converged = 0;
        }
    }
}

#ifndef __SYNTHESIS__
template <typename T, int rowTemplate>
void calConvergence(int nrows, T tol, bool& converged, buffT* bufferPing, buffT* bufferPong) {
#else
template <typename T, int rowTemplate>
void calConvergence(int nrows, T tol, bool& converged, buffT bufferPing[rowTemplate], buffT bufferPong[rowTemplate]) {
#endif
#pragma HLS inline off
#pragma HLS dataflow

    const int wN = (sizeof(T) == 4) ? 16 : 8;
    const int BURST_LENTH = 32;
    const int iteration = (nrows + wN - 1) / wN;
    int extra = iteration * wN - nrows;

    hls::stream<T> subStream[wN];
    // clang-format off
    hls::stream<buffT> pingStrm;
    hls::stream<buffT> pongStrm;
// clang-format on
#pragma HLS stream depth = 16 variable = subStream
#pragma HLS stream depth = 16 variable = pingStrm
#pragma HLS stream depth = 16 variable = pongStrm
#pragma HLS resource variable = subStream core = FIFO_LUTRAM
#pragma HLS resource variable = pingStrm core = FIFO_LUTRAM
#pragma HLS resource variable = pongStrm core = FIFO_LUTRAM
    xf::graph::internal::burstRead2Strm<buffT>(iteration, bufferPing, pingStrm);
    xf::graph::internal::burstRead2Strm<buffT>(iteration, bufferPong, pongStrm);
    vecSub<T, rowTemplate, wN>(nrows, pingStrm, pongStrm, subStream);
    convergenceCal<T, wN>(nrows, tol, converged, subStream);
}

template <typename T>
void calculateMul(
    int nnz, hls::stream<T>& inStrm, hls::stream<T>& cntStrm, hls::stream<float>& weightStrm, hls::stream<T>& mulStrm) {
#pragma HLS inline off
    for (int i = 0; i < nnz; ++i) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
        T tmp1 = inStrm.read();
        T tmp2 = cntStrm.read();
        T tmp3 = weightStrm.read();
        T mul = tmp1 * tmp2 * tmp3;
#ifndef __SYNTHESIS__
// std::cout<<"vector : "<<tmp1<<std::endl;
#endif
        mulStrm.write(mul);
    }
}

template <typename T>
void adderUn(ap_uint<32> distance, T adder, ap_uint<1> flag, hls::stream<T>& splitStrm, T& outSum) {
#pragma HLS inline off
    if ((distance == 0) && (flag == 0)) {
        outSum = adder;
    } else if ((distance == 0) && (flag == 1)) {
        outSum = 0;
    } else if ((distance == 0) && (flag == 1)) {
    } else {
        T tmp = adder;
        const int II1 = (sizeof(T) == 8) ? 6 : 5;
        for (int i = 0; i < distance; ++i) {
#pragma HLS loop_tripcount min = 5 avg = 5 max = 5
#pragma HLS pipeline II = II1
            tmp += splitStrm.read();
        }
        outSum = tmp;
    }
}

// clang-format off
template <typename T, int unrollNm, int widthOr>
void getOffset(int nrows,
                 hls::stream<ap_uint<widthOr> >& offsetStrm,
                 hls::stream<ap_uint<32> >& distStrm2,
                 hls::stream<ap_uint<2> > flagUnStrm[unrollNm]){
// clang-format on
#pragma HLS inline off

    ap_uint<32> counter1[unrollNm];
#pragma HLS array_partition variable = counter1 dim = 0 complete
    ap_uint<32> prev = 0;
    int iteration = (nrows + unrollNm - 1) / unrollNm;
    ap_uint<32> tmpVal[unrollNm];
#pragma HLS array_partition variable = tmpVal dim = 0 complete
    ap_uint<32> valP = prev;
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
                tmpVal[k].range(31, 0) = tmp.range(32 * (k + 1) - 1, 32 * k);

                ap_uint<1> noIOedgeflag = (tmpVal[k] == 0xffffffff) ? 1 : 0;
                ap_uint<1> onlyIedgeflag = tmpVal[k][31];
                ap_uint<2> flag;
                flag[0] = noIOedgeflag;
                flag[1] = onlyIedgeflag;
                flagUnStrm[k].write(flag);
                ap_uint<31> offset = tmpVal[k].range(30, 0);
#ifndef __SYNTHESIS__
// std::cout<<"offset: "<<(int)offset<<std::endl;
#endif

                if (tmpVal[k] == 0xffffffff) {
                    counter1[k] = 0;
                } else if ((i == 0) && (k == 0)) {
                    counter1[k] = offset;
                } else if ((i != 0) && (k == 0)) {
                    counter1[k] = offset - prev;
                } else {
                    counter1[k] = offset - valP;
                }
                distStrm2.write(counter1[k]);
                if (tmpVal[k] != 0xffffffff) {
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
template <int unrollNm>
void dispatchFlag(int nrows,
                 int iteration,
                 int nnz,
                 hls::stream<ap_uint<2> > flagUnStrm[unrollNm],
                 hls::stream<ap_uint<2> > flagUnStrm2[unrollNm]) {
// clang-format on
#pragma HLS inline off
Loop_iter:
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/unrollNm avg = 3700000/unrollNm max = 3700000/unrollNm
    // clang-format on
    Loop_unroll:
        for (int k = 0; k < unrollNm; ++k) {
#pragma HLS loop_tripcount min = unrollNm avg = unrollNm max = unrollNm
            int index = i * unrollNm + k;
            if (index < nrows) {
                ap_uint<2> fflag = flagUnStrm[k].read();
                flagUnStrm2[k].write(fflag);
            }
        }
    }
}

// clang-format off
template <typename T, int unrollNm, int unrollBin>
void dispatchPR(int nrows,
                 int nnz,
                 hls::stream<ap_uint<32> >& distStrm,
                 hls::stream<T>& mulStrm,
                 hls::stream<T> tmpStrm[unrollNm],
                 hls::stream<ap_uint<1> > distEStrm[unrollNm]){
// clang-format on
#pragma HLS inline off
    ap_uint<unrollBin> k = 0;
    int cntRow = 0;
    int distance = distStrm.read();
    while (cntRow < nrows) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
        if ((cntRow == nrows - 1) && (distance == 0)) {
            distEStrm[k].write(1);
            cntRow++;
        } else if (distance == 0) {
            distEStrm[k].write(1);
            k++;
            distance = distStrm.read();
            cntRow++;
        } else {
            T tmp = mulStrm.read();
            tmpStrm[k].write(tmp);
            distEStrm[k].write(0);
            distance--;
        }
    }
}

// clang-format off
template <typename T, int unrollNm>
void adderWrapper(int k,
                  int nrows,
                  int iteration,
                  T adder,
                  hls::stream<ap_uint<1> >& distEStrm,
                  hls::stream<ap_uint<2> >& flagStrm,
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
            ap_uint<2> flag = flagStrm.read();
            T outSum = (flag[0] || flag[1]) ? 0 : adder;

            while (!distEStrm.read()) {
#pragma HLS loop_tripcount min = 1 avg = 5 max = 5
#pragma HLS pipeline II = 5
                if (flag[1]) {
                    tmpStrm.read();
                    outSum = 1.0;
                } else {
                    outSum += tmpStrm.read();
                }
            }
            outStrm.write(outSum);
#ifndef __SYNTHESIS__
            std::cout << "outSum: " << outSum << std::endl;
#endif
        }
    }
}

// clang-format off
template <typename T, int unrollNm>
void adderPart2(int nrows,
                int iteration,
                T adder,
                hls::stream<ap_uint<1> > distEStrm[unrollNm],
                hls::stream<ap_uint<2> > flagUnStrm2[unrollNm],
                hls::stream<T> tmpStrm[unrollNm],
                hls::stream<T> outStrm[unrollNm]) {
// clang-format on
#pragma HLS inline off
#pragma HLS dataflow

Loop_adder2:
    for (int k = 0; k < unrollNm; ++k) {
#pragma HLS unroll factor = unrollNm
        adderWrapper<T, unrollNm>(k, nrows, iteration, adder, distEStrm[k], flagUnStrm2[k], tmpStrm[k], outStrm[k]);
    }
}

// clang-format off
template <typename T, int rowTemplate, int unrollNm, int unrollBin>
void combineStrm512(int nrows, hls::stream<T> outStrm[unrollNm], hls::stream<buffT>& outStrm2) {
// clang-format on
#pragma HLS inline off
    ap_uint<unrollBin> cnt = 0;
    const int size = sizeof(T) * 8;
    const int size2 = sizeof(T) * 4;
    const int wN = (sizeof(T) == 4) ? 16 : 8;
    const int iteration = (nrows + wN - 1) / wN;
    buffT tmp = 0;
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
                outStrm2.write(tmp);
            }
        }
    }
}

#ifndef __SYNTHESIS__
template <typename T, int rowTemplate>
void writeOut(int nrows, hls::stream<buffT>& outStrm2, buffT* bufferWrite) {
#else
template <typename T, int rowTemplate>
void writeOut(int nrows, hls::stream<buffT>& outStrm2, buffT bufferWrite[rowTemplate]) {
#endif
#pragma HLS inline off
    const int wN = (sizeof(T) == 4) ? 16 : 8;
    const int iteration = (nrows + wN - 1) / wN;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        bufferWrite[i] = outStrm2.read();
    }
}

template <typename T>
void transfer(int nnz,
              hls::stream<ap_uint<8 * sizeof(T)> >& cntStrm2,
              hls::stream<ap_uint<8 * sizeof(T)> >& pingStrm2,
              hls::stream<T>& cntStrm,
              hls::stream<T>& pingStrm) {
#pragma HLS inline off
    const int widthT = 8 * sizeof(T);
    for (int i = 0; i < nnz; ++i) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1

        calc_degree::f_cast<T> constVal;
        calc_degree::f_cast<T> pingVal;
        constVal.i = cntStrm2.read();
        pingVal.i = pingStrm2.read();
        if (sizeof(T) == 8) {
            cntStrm.write(constVal.f);
            pingStrm.write(pingVal.f);
        } else {
            cntStrm.write(constVal.f);
            pingStrm.write(pingVal.f);
        }
    }
}

template <typename T>
void transfer(int nnz, hls::stream<ap_uint<8 * sizeof(T)> >& inputStrm, hls::stream<T>& outputStrm) {
#pragma HLS inline off
    const int widthT = 8 * sizeof(T);
    for (int i = 0; i < nnz; ++i) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
        calc_degree::f_cast<T> tmp;
        tmp.i = inputStrm.read();
        outputStrm.write(tmp.f);
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
    T adder,
    xf::common::utils_hw::
        cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    buffT cntVal[rowTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    buffT buffPing[rowTemplate],
    buffT buffPong[rowTemplate]) {
#pragma HLS inline off
#pragma HLS dataflow
    const int widthT = sizeof(T) * 8;
    const int iteration = (sizeof(T) == 8) ? (nrows + 7) / 8 : (nrows + 15) / 16;
    // clang-format off
    hls::stream<ap_uint<32> >       indiceStrm("indiceStrm");
#pragma HLS resource     variable = indiceStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = indiceStrm depth = 16

    hls::stream<ap_uint<32> >       weightStrm("weightStrm");
#pragma HLS resource     variable = weightStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = weightStrm depth = 16

    hls::stream<ap_uint<widthT> >   pingStrm("pingStrm");
#pragma HLS resource     variable = pingStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = pingStrm depth = 16

    hls::stream<ap_uint<widthT> >   cntStrm("cntStrm");
#pragma HLS resource     variable = cntStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = cntStrm depth = 16

    hls::stream<T>                  pingStrmT("pingStrmT");
#pragma HLS resource     variable = pingStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = pingStrmT depth = 16

    hls::stream<T>                  cntStrmT("cntStrmT");
#pragma HLS resource     variable = cntStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = cntStrmT depth = 16

    hls::stream<float>              weightStrmT("weightStrmT");
#pragma HLS resource     variable = weightStrmT core = FIFO_LUTRAM
#pragma HLS stream       variable = weightStrmT depth = 16

    hls::stream<T>                  mulStrm("mulStrm");
#pragma HLS resource     variable = mulStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = mulStrm depth = 16

    hls::stream<ap_uint<widthOr> >  offsetStrm("offsetStrm");
#pragma HLS resource     variable = offsetStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = offsetStrm depth = 16

    hls::stream<ap_uint<32> >       distStrm;
#pragma HLS resource     variable = distStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = distStrm depth = 16

    hls::stream<ap_uint<2> >        flagUnStrm[unrollNm];
#pragma HLS resource     variable = flagUnStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = flagUnStrm depth = 16

    hls::stream<T>                  tmpStrm[unrollNm];
#pragma HLS resource     variable = tmpStrm core = FIFO_BRAM
#pragma HLS stream       variable = tmpStrm depth = 4096

    hls::stream<ap_uint<1> >        distEStrm[unrollNm];
#pragma HLS resource     variable = distEStrm core = FIFO_LUTRAM
#pragma HLS stream       variable = distEStrm depth = 4096

    hls::stream<ap_uint<2> >        flagUnStrm2[unrollNm];
#pragma HLS resource     variable = flagUnStrm2 core = FIFO_LUTRAM
#pragma HLS stream       variable = flagUnStrm2 depth = 4096

    hls::stream<T>                  outStrm[unrollNm];
#pragma HLS resource     variable = outStrm core = FIFO_BRAM
#pragma HLS stream       variable = outStrm depth = 1024

    hls::stream<buffT>              outStrm2;
#pragma HLS resource     variable = outStrm2 core = FIFO_LUTRAM
#pragma HLS stream       variable = outStrm2 depth = 16
    // clang-format on

    xf::common::utils_hw::axiToStream<BURST_LENTH, _Width, ap_uint<32> >(indices, nnz, indiceStrm);
    xf::common::utils_hw::axiToStream<BURST_LENTH, _Width, ap_uint<32> >(weight, nnz, weightStrm);
    // ap_uint<64>, 1, 1, 16, 32, 0, 0, 0
    cache0.readOnly(nnz, buffPing, cntVal, indiceStrm, cntStrm, pingStrm);
    transfer<T>(nnz, cntStrm, pingStrm, cntStrmT, pingStrmT);
    transfer<float>(nnz, weightStrm, weightStrmT);
    calculateMul<T>(nnz, pingStrmT, cntStrmT, weightStrmT, mulStrm);
    // clang-format off
    xf::graph::internal::burstRead2Strm<ap_uint<widthOr> >(iteration, order, offsetStrm);
    // clang-format on
    getOffset<T, unrollNm, widthOr>(nrows, offsetStrm, distStrm, flagUnStrm);

    dispatchPR<T, unrollNm, unrollBin>(nrows, nnz, distStrm, mulStrm, tmpStrm, distEStrm);
    dispatchFlag<unrollNm>(nrows, iteration, nnz, flagUnStrm, flagUnStrm2);
    adderPart2<T, unrollNm>(nrows, iteration, adder, distEStrm, flagUnStrm2, tmpStrm, outStrm);
    combineStrm512<T, rowTemplate, unrollNm, unrollBin>(nrows, outStrm, outStrm2);
    writeOut<T, rowTemplate>(nrows, outStrm2, buffPong);
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
    ap_uint<1>& share,
    T adder,
    xf::common::utils_hw::
        cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    buffT cntVal[rowTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    buffT buffPing[rowTemplate],
    buffT buffPong[rowTemplate]) {
#pragma HLS inline off
    cache0.initDualOffChip();
    if (share) {
        dataFlowPart<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                     dataOneLine, addrWidth, usURAM>(nrows, nnz, adder, cache0, indices, weight, cntVal, order,
                                                     buffPing, buffPong);
    } else {
        dataFlowPart<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                     dataOneLine, addrWidth, usURAM>(nrows, nnz, adder, cache0, indices, weight, cntVal, order,
                                                     buffPong, buffPing);
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
    ap_uint<1>& share,
    T adder,
    xf::common::utils_hw::
        cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM, usURAM>& cache0,
    buffT indices[NNZTemplate],
    buffT weight[NNZTemplate],
    buffT cntVal[rowTemplate],
    ap_uint<widthOr> order[rowTemplate / unrollNm],
    buffT buffPing[rowTemplate],
    buffT buffPong[rowTemplate],
    T tol,
    bool& converged) {
#pragma HLS inline off
    dataFlowWrapper<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow, groupUramPart,
                    dataOneLine, addrWidth, usURAM>(nrows, nnz, share, adder, cache0, indices, weight, cntVal, order,
                                                    buffPing, buffPong);
    calConvergence<T, rowTemplate>(nrows, tol, converged, buffPing, buffPong);
}

// clang-format off
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
    ap_uint<32> tVal = 0xffffffff;
    ap_uint<1> onlyIedgeflag = 1;
    int realNrows = nrows;
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
                    orderT[k] = tVal;
                } else if ((cntCSR.f == 0.0) && (cntCSC != 0)) {
                    pongT[k] = 1.0;
                    orderT[k][31] = onlyIedgeflag;
                    orderT[k].range(sizeT - 2, 0) = (ap_uint<31>)tmpCSC;
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
#ifndef __SYNTHESIS__
                    std::cout << "const:" << tTmp2.f << std::endl;
#endif
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
    ap_uint<32> tVal = 0xffffffff;
    int realNrows = nrows;
    const int sizeT = 64;
    const int sizeT2 = 32;
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
                if ((cntCSR.f == 0.0) && (cntCSC == 0)) {
                    pongT[k] = 0;
                    orderT[k] = tVal;
                } else if ((cntCSR.f == 0.0) && (cntCSC != 0)) {
                    pongT[k] = 1.0;
                    orderT[k][31] = onlyIedgeflag;
                    orderT[k].range(30, 0) = (ap_uint<31>)tmpCSC;
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
#ifndef __SYNTHESIS__
                    std::cout << "const:" << tTmp2.f << std::endl;
#endif
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
              hls::stream<buffT>& csrDegree,
              hls::stream<buffT>& cscOffset,
              hls::stream<buffT>& pongStrm,
              hls::stream<buffT>& cntStrm,
              hls::stream<ap_uint<widthOr> >& orderStrm) {
// clang-format on
#pragma HLS inline off
    const int size0 = sizeof(T);
    if (size0 == 4) {
        preWrite32<T>(nrows, alpha, randomProbability, csrDegree, cscOffset, pongStrm, cntStrm, orderStrm);
    } else if (size0 == 8) {
        preWrite64<T>(nrows, alpha, randomProbability, csrDegree, cscOffset, pongStrm, cntStrm, orderStrm);
    }
}

template <typename T>
void combine512(int nrows, hls::stream<buffT> pingStrm[2], hls::stream<buffT>& pingStrm2) {
#pragma HLS inline off
    int iteration = (nrows + 15) / 16;
    int cnt = 0;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/16 avg = 3700000/16 max = 3700000/16
// clang-format on
#pragma HLS pipeline II = 1
        if (sizeof(T) == 4) {
            pingStrm2.write(pingStrm[0].read());
        } else if (sizeof(T) == 8) {
            if (cnt * 8 < nrows) {
                pingStrm2.write(pingStrm[0].read());
            }
            if (cnt * 8 + 8 < nrows) {
                pingStrm2.write(pingStrm[1].read());
            }
            cnt++;
        }
        cnt++;
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
void writeOutDDR(int nrows,
                 hls::stream<buffT>& cntStrm,
                 hls::stream<buffT>& pongStrm,
                 buffT cntValFull[rowTemplate],
                 buffT buffPong[rowTemplate]) {
#pragma HLS inline off
    const int iteration = (sizeof(T) == 8) ? (nrows + 7) / 8 : (nrows + 15) / 16;
    for (int i = 0; i < iteration; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = 3700000/8 avg = 3700000/8 max = 3700000/8
// clang-format on
#pragma HLS pipeline II = 1
        buffPong[i] = pongStrm.read();
        cntValFull[i] = cntStrm.read();
    }
}

template <typename T, int BURST_LENTH, int rowTemplate, int NNZTemplate, int widthOr>
void initDDR(int nrows,
             int nnz,
             T alpha,
             T randomProbability,
             buffT degreeCSR[rowTemplate],
             buffT offsetCSC[rowTemplate],
             buffT cntValFull[rowTemplate],
             buffT buffPong[rowTemplate],
             ap_uint<widthOr> orderUnroll[rowTemplate]) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<buffT> csrDegree("csrDegree");
#pragma HLS stream depth = 8 variable = csrDegree
    hls::stream<buffT> cscOffset("cscOffset");
#pragma HLS stream depth = 8 variable = cscOffset
    hls::stream<buffT> pongStrm;
#pragma HLS stream depth = 8 variable = pongStrm
    hls::stream<buffT> cntStrm;
#pragma HLS stream depth = 8 variable = cntStrm
    hls::stream<ap_uint<widthOr> > orderStrm;
#pragma HLS stream depth = 8 variable = orderStrm
#pragma HLS resource variable = csrDegree core = FIFO_LUTRAM
#pragma HLS resource variable = cscOffset core = FIFO_LUTRAM
#pragma HLS resource variable = pongStrm core = FIFO_LUTRAM
#pragma HLS resource variable = cntStrm core = FIFO_LUTRAM
#pragma HLS resource variable = orderStrm core = FIFO_LUTRAM
    const int wN = 16;
    const int iteration = (nrows + wN - 1) / wN;
    const int extra = iteration * wN - nrows;
    xf::graph::internal::axiToCharStream<BURST_LENTH, _Width, buffT>(degreeCSR, csrDegree, 4 * (nrows + extra));
    xf::graph::internal::axiToCharStream<BURST_LENTH, _Width, buffT>(offsetCSC, cscOffset, 4 * (nrows + extra), 4);

    preWrite<T, widthOr>(nrows, alpha, randomProbability, csrDegree, cscOffset, pongStrm, cntStrm, orderStrm);
    writeOutDDR<T, rowTemplate>(nrows, cntStrm, pongStrm, cntValFull, buffPong);
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
                  // buffT* pagerank,
                  buffT* buffPing,
                  buffT* buffPong,
                  ap_uint<widthOr>* order,
                  buffT* indices,
                  buffT* weight,
                  buffT* cntVal,
                  int* resultInfo,
                  T alpha = 0.85,
                  T tolerance = 1e-4,
                  int maxIter = 200) {
    const int dataUramNmBin = 0;
    const int dataOneLine = 1 << dataOneLineBin;  // double 8 : float 16
    const int uramRow = 1 << uramRowBin;          // 4096
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

    int iterator = 0;

    xf::common::utils_hw::cache<ap_uint<sizeof(T) * 8>, uramRow, groupUramPart, dataOneLine, addrWidth, usURAM, usURAM,
                                usURAM>
        cache0;
    while (!converged && iterator < maxIt) {
#pragma HLS loop_tripcount min = 16 avg = 16 max = 16

        iterator++;
        converged = 1;

        internal::pagerank::dataFlowTop<BURST_LENTH, T, rowTemplate, NNZTemplate, unrollNm, unrollBin, widthOr, uramRow,
                                        groupUramPart, dataOneLine, addrWidth, usURAM>(
            nrows, nnz, share, adder, cache0, indices, weight, cntVal, order, buffPing, buffPong, tol, converged);
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
 *
 * @tparam T date type of pagerank, double or float
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2UNROLL log2 of unroll number, due to DDR limit, best LOG2UNROLL is 3
 * @tparam WIDTHOR order array bandwidth, it's 256 in our case
 * @tparam LOG2CACHEDEPTH log2(cache depth), the onchip memory for cache is 512 bit x CACHEDEPTH (512 bit x
 * 2^LOG2CACHEDEPTH)
 * @tparam LOG2DATAPERCACHELINECORE param for module pageRankCore, log2 of number of data in one 512bit (64 byte), for
 * double, it's log2(64/sizeof(double)) = 3, for float, it's log2(64/sizeof(float)) = 4
 * @tparam LOG2DATAPERCACHELINEDEGREE param for module calduDegree, log2 of number of data in one 512bit (64 byte), for
 * double, it's log2(64/sizeof(double)) = 3, for float, it's log2(64/sizeof(float)) = 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 *
 * @param numVertex CSR/CSC data offsets number
 * @param numEdge CSR/CSC data indices number
 * @param degreeCSR temporary internal degree value
 * @param offsetCSC CSR/CSC data offset array
 * @param indexCSC CSR/CSC data indice array
 * @param weightCSC CSR/CSC data weight array, support type float
 * @param cntValFull temporary internal initialized mulplier values, length equals to numVertex
 * @param buffPing ping array to keep temporary pagerank value
 * @param buffPong pong array to keep temporary pagerank value
 * @param orderUnroll temporary internal order array to keep initialized offset values
 * @param resultInfo The output information. resultInfo[0] is isResultinPong, resultInfo[1] is iterations.
 * @param randomProbability initial PR value, normally 1.0 or 1.0/numVertex
 * @param alpha damping factor, normally 0.85
 * @param tolerance converge tolerance
 * @param numIter max iteration
 */

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
                 ap_uint<512>* degreeCSR,
                 ap_uint<512>* offsetCSC,
                 ap_uint<512>* indexCSC,
                 ap_uint<512>* weightCSC,
                 ap_uint<512>* cntValFull,
                 ap_uint<512>* buffPing,
                 ap_uint<512>* buffPong,
                 ap_uint<WIDTHOR>* orderUnroll,
                 int* resultInfo,
                 T randomProbability = 1.0,
                 T alpha = 0.85,
                 T tolerance = 1e-4,
                 int numIter = 200) {
    const int BURST_LENTH = 32;
    xf::graph::calcuWeightedDegree<MAXVERTEX, MAXEDGE, LOG2CACHEDEPTH, LOG2DATAPERCACHELINEDEGREE, RAMTYPE>(
        numVertex, numEdge, indexCSC, weightCSC, degreeCSR);
    internal::pagerank::initDDR<T, BURST_LENTH, MAXVERTEX, MAXEDGE, WIDTHOR>(
        numVertex, numEdge, alpha, randomProbability, degreeCSR, offsetCSC, cntValFull, buffPong, orderUnroll);
    pageRankCore<T, MAXVERTEX, MAXEDGE, LOG2UNROLL, WIDTHOR, LOG2CACHEDEPTH, LOG2DATAPERCACHELINECORE, RAMTYPE>(
        numVertex, numEdge, buffPing, buffPong, orderUnroll, indexCSC, weightCSC, cntValFull, resultInfo, alpha,
        tolerance, numIter);
}

} // namespace graph
} // namespace xf
#endif //#ifndef VT_GRAPH_PR_H
