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

#ifndef _XF_GRAPH_LABEL_PROPAGATION_HPP_
#define _XF_GRAPH_LABEL_PROPAGATION_HPP_

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "L2_utils.hpp"
#include "xf_fintech/xoshiro128.hpp"
#include "xf_database/merge_sort.hpp"
#include "hash_max_freq.hpp"

namespace xf {
namespace graph {
namespace internal {
namespace label_propagation {

template <typename DT>
void offsetCtrlIndex(int vertexNum,
                     hls::stream<DT>& offsetStrm,
                     hls::stream<DT>& indexStrm,
                     hls::stream<DT>& indexOutStrm,
                     hls::stream<bool>& indexEndStrm) {
    DT begin = 0;
    DT end = 0;
    for (int i = 0; i < vertexNum; i++) {
#pragma HLS loop_tripcount max = 1000 min = 1000
        end = offsetStrm.read();
        for (int j = begin; j < end; j++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
            indexOutStrm.write(indexStrm.read());
            indexEndStrm.write(false);
        }
        begin = end;
        indexEndStrm.write(true);
    }
}

template <typename DT>
void mergeSortWrapper(int num,
                      hls::stream<DT>& in1Strm,
                      hls::stream<bool>& inEnd1Strm,
                      hls::stream<DT>& in2Strm,
                      hls::stream<bool>& inEnd2Strm,
                      hls::stream<DT>& outStrm,
                      hls::stream<bool>& outEndStrm) {
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 1000 min = 1000
        xf::database::mergeSort<DT>(in1Strm, inEnd1Strm, in2Strm, inEnd2Strm, outStrm, outEndStrm, 1);
    }
}

template <typename DT>
void mostFreqLabel(int vertex,
                   hls::stream<DT>& labelOrderStrm,
                   hls::stream<bool>& labelOrderEndStrm,
                   hls::stream<DT>& mostLabelStrm) {
    DT maxLabel = vertex;
    while (!labelOrderEndStrm.read()) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        maxLabel = labelOrderStrm.read();
    }
    mostLabelStrm.write(maxLabel);
}

template <typename DT, typename uint512>
void sortImpl(int vertexNum,
              hls::stream<DT>& labelStrm,
              hls::stream<DT>& rngStrm,
              hls::stream<bool>& labelEndStrm,
              uint512* pingHashBuf,
              uint512* pongHashBuf,
              hls::stream<DT>& labelOrderStrm,
              hls::stream<bool>& labelOrderEndStrm) {
    hls::stream<DT> outDataStrm;
    for (int i = 0; i < vertexNum; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
        hashGroupAggregate<32, 11, 512, 32, 32>(labelStrm, rngStrm, labelEndStrm, pingHashBuf, pongHashBuf,
                                                labelOrderStrm, labelOrderEndStrm);
    }
}

template <typename DT>
void labelSelect(int vertexNum,
                 hls::stream<DT>& labelOrderStrm,
                 hls::stream<bool>& labelOrderEndStrm,
                 hls::stream<DT>& mostLabelStrm) {
    for (int i = 0; i < vertexNum; i++) {
#pragma HLS loop_tripcount max = 1000 min = 1000
        mostFreqLabel<DT>(i, labelOrderStrm, labelOrderEndStrm, mostLabelStrm);
    }
}
template <typename DT, typename uint512, int K, int W, typename RNG>
void labelProcess(int vertexNum,
                  hls::stream<DT>& rngStrm,
                  hls::stream<DT>& labelStrm,
                  hls::stream<bool>& labelEndStrm,
                  uint512* pingHashBuf,
                  uint512* pongHashBuf,
                  hls::stream<DT>& mostLabelStrm) {
#pragma HLS dataflow
    const int strmDepth = 16;
    hls::stream<DT> labelOrderStrm("labelOrderStrm");
#pragma HLS stream variable = labelOrderStrm depth = strmDepth
    hls::stream<bool> labelOrderEndStrm("labelOrderEndStrm");
#pragma HLS stream variable = labelOrderEndStrm depth = strmDepth
    sortImpl<DT>(vertexNum, labelStrm, rngStrm, labelEndStrm, pingHashBuf, pongHashBuf, labelOrderStrm,
                 labelOrderEndStrm);
    labelSelect<DT>(vertexNum, labelOrderStrm, labelOrderEndStrm, mostLabelStrm);
}

template <typename RNG, typename DT>
void rngProcess(int vertexNum, RNG& rng, hls::stream<DT>& rngStrm) {
    for (int i = 0; i < vertexNum; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline ii = 1
        rngStrm.write(rng.next());
    }
}

template <typename DT, typename uint512, int K, int W, typename RNG>
void lpCoreImpl(int edgeNum,
                int vertexNum,
                uint512* indexCSR,
                uint512* offsetCSR,
                uint512* indexCSC,
                uint512* offsetCSC,
                uint512* pingHashBuf,
                uint512* pongHashBuf,
                uint512* labelIn,
                uint512* labelOut,
                RNG& rng) {
#pragma HLS dataflow
    const int strmDepth = 16;
    hls::stream<DT> indexCSRStrm("indexCSRStrm");
#pragma HLS stream variable = indexCSRStrm depth = strmDepth
    hls::stream<DT> indexCSCStrm("indexCSCStrm");
#pragma HLS stream variable = indexCSCStrm depth = strmDepth
    hls::stream<DT> offsetCSRStrm("offsetCSRStrm");
#pragma HLS stream variable = offsetCSRStrm depth = strmDepth
    hls::stream<DT> offsetCSCStrm("offsetCSCStrm");
#pragma HLS stream variable = offsetCSCStrm depth = strmDepth
    hls::stream<DT> indexCSROutStrm("indexCSROutStrm");
#pragma HLS stream variable = indexCSROutStrm depth = strmDepth
    hls::stream<bool> indexCSREndStrm("indexCSREndStrm");
#pragma HLS stream variable = indexCSREndStrm depth = strmDepth
    hls::stream<DT> indexCSCOutStrm("indexCSCOutStrm");
#pragma HLS stream variable = indexCSCOutStrm depth = strmDepth
    hls::stream<bool> indexCSCEndStrm("indexCSCEndStrm");
#pragma HLS stream variable = indexCSCEndStrm depth = strmDepth
    hls::stream<DT> labelAddrStrm("labelAddrStrm");
#pragma HLS stream variable = labelAddrStrm depth = strmDepth
    hls::stream<DT> labelStrm("labelStrm");
#pragma HLS stream variable = labelStrm depth = strmDepth
    hls::stream<bool> labelEndStrm("labelEndStrm");
#pragma HLS stream variable = labelEndStrm depth = 1024
    hls::stream<DT> mostLabelStrm("mostLabelStrm");
#pragma HLS stream variable = mostLabelStrm depth = strmDepth
    hls::stream<uint512> labelOutStrm("labelOutStrm");
#pragma HLS stream variable = labelOutStrm depth = strmDepth
#pragma HLS resource variable = labelOutStrm core = FIFO_LUTRAM
    hls::stream<DT> rngStrm("rngStrm");
#pragma HLS stream variable = rngStrm depth = strmDepth
/***************************************************/
#ifndef __SYNTHESIS__
    std::cout << "burstReadSplit2Strm ing\n";
#endif
    burstReadSplit2Strm<DT, uint512, K, W>(edgeNum, 0, indexCSR, indexCSRStrm);
    burstReadSplit2Strm<DT, uint512, K, W>(edgeNum, 0, indexCSC, indexCSCStrm);
    burstReadSplit2Strm<DT, uint512, K, W>(vertexNum + 1, 1, offsetCSR, offsetCSRStrm);
    burstReadSplit2Strm<DT, uint512, K, W>(vertexNum + 1, 1, offsetCSC, offsetCSCStrm);

#ifndef __SYNTHESIS__
    std::cout << "offsetCtrlIndex ing\n";
#endif
    offsetCtrlIndex<DT>(vertexNum, offsetCSRStrm, indexCSRStrm, indexCSROutStrm, indexCSREndStrm);
    offsetCtrlIndex<DT>(vertexNum, offsetCSCStrm, indexCSCStrm, indexCSCOutStrm, indexCSCEndStrm);

#ifndef __SYNTHESIS__
    std::cout << "mergeSortWrapper ing\n";
#endif
    mergeSortWrapper<DT>(vertexNum, indexCSROutStrm, indexCSREndStrm, indexCSCOutStrm, indexCSCEndStrm, labelAddrStrm,
                         labelEndStrm);
    addrReadData<DT, uint512, K, W>(edgeNum * 2, labelIn, labelAddrStrm, labelStrm);

    rngProcess<RNG, DT>(edgeNum * 2, rng, rngStrm);
#ifndef __SYNTHESIS__
    std::cout << "labelProcess ing\n";
#endif
    labelProcess<DT, uint512, K, W, RNG>(vertexNum, rngStrm, labelStrm, labelEndStrm, pingHashBuf, pongHashBuf,
                                         mostLabelStrm);
#ifndef __SYNTHESIS__
    std::cout << "combineStrm ing\n";
#endif
    combineStrm<DT, uint512, K, W>(vertexNum, mostLabelStrm, labelOutStrm);
    burstWrite2Strm<uint512>((vertexNum + K - 1) / K, labelOutStrm, labelOut);
}

template <typename DT, int K>
void ascendNum2Strm(int len, int offset, hls::stream<DT>& numStrm) {
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline
        numStrm.write(i * K + offset);
    }
}

template <typename DT, typename uint512, int K, int W>
void combineNTo1Strm(int len, hls::stream<DT> numStrm[K], hls::stream<uint512>& combineNumStrm) {
    uint512 tmp;
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 1000 min = 1000
        for (int k = 0; k < K; k++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
            tmp.range(k * W + W - 1, k * W) = numStrm[k].read();
        }
        combineNumStrm.write(tmp);
    }
}
template <typename DT, typename uint512, int K, int W>
void labelInit(int vertexNum, uint512* labelArr) {
#pragma HLS dataflow
    hls::stream<DT> numStrm[K];
    hls::stream<uint512> combineNumStrm;
    int len = (vertexNum + K - 1) / K;
    for (int i = 0; i < K; i++) {
#pragma HLS unroll
        ascendNum2Strm<DT, K>(len, i, numStrm[i]);
    }
    combineNTo1Strm<DT, uint512, K, W>(len, numStrm, combineNumStrm);
    burstWrite2Strm<uint512>(len, combineNumStrm, labelArr);
}
} // namespace label_propagation
} // namespace internal

/**
 * @brief labelPropagation the label propagation algorithm is implemented
 *
 * @param numEdge edge number of the graph
 * @param numVertex vertex number of the graph
 * @param numIter iteration number
 * @param indexCSR column index of CSR format
 * @param offsetCSR row offset of CSR format
 * @param indexCSC row index of CSC format
 * @param offsetCSC column of CSC format
 * @param labelPing label ping buffer
 * @param labelPong label pong buffer
 */
inline void labelPropagation(int numEdge,
                             int numVertex,
                             int numIter,
                             ap_uint<512>* offsetCSR,
                             ap_uint<512>* indexCSR,
                             ap_uint<512>* offsetCSC,
                             ap_uint<512>* indexCSC,
                             ap_uint<512>* pingHashBuf,
                             ap_uint<512>* pongHashBuf,
                             ap_uint<512>* labelPing,
                             ap_uint<512>* labelPong) {
    internal::label_propagation::labelInit<ap_uint<32>, ap_uint<512>, 16, 32>(numVertex, labelPing);
    xf::fintech::XoShiRo128PlusPlus rng;
    unsigned int seed[4] = {1, 2, 3, 4};
    rng.init(seed);
    rng.jump();
loop_numIter:
    for (int i = 0; i < numIter; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
        if (i % 2 == 0)
            internal::label_propagation::lpCoreImpl<ap_uint<32>, ap_uint<512>, 16, 32, xf::fintech::XoShiRo128PlusPlus>(
                numEdge, numVertex, indexCSR, offsetCSR, indexCSC, offsetCSC, pingHashBuf, pongHashBuf, labelPing,
                labelPong, rng);
        else
            internal::label_propagation::lpCoreImpl<ap_uint<32>, ap_uint<512>, 16, 32, xf::fintech::XoShiRo128PlusPlus>(
                numEdge, numVertex, indexCSR, offsetCSR, indexCSC, offsetCSC, pingHashBuf, pongHashBuf, labelPong,
                labelPing, rng);
    }
}

} // namespace graph
} // namespace xf
#endif
