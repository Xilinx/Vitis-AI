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

#include "kernel_pagerank.hpp"

#ifdef KERNEL0

#if (CHANNEL_NUM == 6)

extern "C" void kernel_pagerank_0(int nrows,
                                  int nnz,
                                  DT alpha,
                                  DT tolerance,
                                  int maxIter,
                                  int nsource,
                                  ap_uint<32>* sourceID,
                                  buffType* offsetCSC,
                                  buffType* indiceCSC,
                                  buffType* weightCSC,
                                  buffType* degreeCSR,
                                  buffType* cntValFull0,
                                  buffType* buffPing0,
                                  buffType* buffPong0,
                                  buffType* cntValFull1,
                                  buffType* buffPing1,
                                  buffType* buffPong1,
                                  buffType* cntValFull2,
                                  buffType* buffPing2,
                                  buffType* buffPong2,
                                  buffType* cntValFull3,
                                  buffType* buffPing3,
                                  buffType* buffPong3,
                                  buffType* cntValFull4,
                                  buffType* buffPing4,
                                  buffType* buffPong4,
                                  buffType* cntValFull5,
                                  buffType* buffPing5,
                                  buffType* buffPong5,
                                  int* resultInfo,
                                  ap_uint<widthOr>* orderUnroll) {
    const int depthOffset = depOffset;
    const int depthVertex = depVertex;
    const int depthEdge = depEdge;
    const int cacheDepthBin =
        (CHANNEL_NUM == 6) ? 12 : 14; // cache line depth in Binary per channel, total (1<<cacheDepthBin)*channels
    const int AxiLatency = (cacheDepthBin == 0) ? 32 : 64;

    const int OffChipLatency = 36; // 64;

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = sourceID latency = \
    OffChipLatency num_read_outstanding = 16 max_read_burst_length = 32 depth = depthOffset
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = offsetCSC latency = \
    OffChipLatency num_read_outstanding = 16 max_read_burst_length = 32 depth = depthOffset
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = orderUnroll latency =          \
    OffChipLatency num_read_outstanding = 16 max_read_burst_length = 32 num_write_outstanding = \
        16 max_write_burst_length = 32 depth = depOffset

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = indiceCSC latency = \
    OffChipLatency num_read_outstanding = 16 max_read_burst_length = 32 depth = depthEdge
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem4 port = weightCSC latency = \
    OffChipLatency num_read_outstanding = 16 max_read_burst_length = 32 depth = depthEdge
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem5 port = degreeCSR latency = AxiLatency num_read_outstanding = \
    32 max_read_burst_length = 2 num_write_outstanding = 32 max_write_burst_length = 2 depth = depthOffset

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem6 port = cntValFull0 latency =         \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem7 port = buffPing0 latency =           \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem8 port = buffPong0 latency =           \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem9 port = cntValFull1 latency =         \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem10 port = buffPing1 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem11 port = buffPong1 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem12 port = cntValFull2 latency =        \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem13 port = buffPing2 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem14 port = buffPong2 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem15 port = cntValFull3 latency =        \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem16 port = buffPing3 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem17 port = buffPong3 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem18 port = cntValFull4 latency =        \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem19 port = buffPing4 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem20 port = buffPong4 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem21 port = cntValFull5 latency =        \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem22 port = buffPing5 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem23 port = buffPong5 latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem7 port = resultInfo latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = 2

#pragma HLS INTERFACE s_axilite port = sourceID bundle = control
#pragma HLS INTERFACE s_axilite port = offsetCSC bundle = control
#pragma HLS INTERFACE s_axilite port = indiceCSC bundle = control
#pragma HLS INTERFACE s_axilite port = weightCSC bundle = control
#pragma HLS INTERFACE s_axilite port = degreeCSR bundle = control
#pragma HLS INTERFACE s_axilite port = cntValFull0 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing0 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong0 bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull1 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing1 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong1 bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull2 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing2 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong2 bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull3 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing3 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong3 bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull4 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing4 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong4 bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull5 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing5 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong5 bundle = control

#pragma HLS INTERFACE s_axilite port = resultInfo bundle = control
#pragma HLS INTERFACE s_axilite port = orderUnroll bundle = control

#pragma HLS INTERFACE s_axilite port = nsource bundle = control
#pragma HLS INTERFACE s_axilite port = nnz bundle = control
#pragma HLS INTERFACE s_axilite port = nrows bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = tolerance bundle = control
#pragma HLS INTERFACE s_axilite port = maxIter bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    DT randomProbability = 1.0; // the init of PR, normally use 1.0 or 1.0/nVertex
    const int widthT = (BITS_IN_BYTE * sizeof(DT));
    const int dataOneLineBin = unrollBin;       // data numbers in Binary of one buffType
    const int dataOneLineBin2 = floatUnrollBin; // data numbers in Binary of one buffType
    const int usURAM =
        (cacheDepthBin == 0) ? 0 : 1; // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::pageRankTop<DT, maxVertex, maxEdge, unrollBin, widthOr, cacheDepthBin, dataOneLineBin, dataOneLineBin2,
                           usURAM>(nrows, nnz, nsource, sourceID, degreeCSR, offsetCSC, indiceCSC, weightCSC,
                                   cntValFull0, buffPing0, buffPong0, cntValFull1, buffPing1, buffPong1, cntValFull2,
                                   buffPing2, buffPong2, cntValFull3, buffPing3, buffPong3, cntValFull4, buffPing4,
                                   buffPong4, cntValFull5, buffPing5, buffPong5, orderUnroll, resultInfo,
                                   randomProbability, alpha, tolerance, maxIter);
}

#else

extern "C" void kernel_pagerank_0(int nrows,
                                  int nnz,
                                  DT alpha,
                                  DT tolerance,
                                  int maxIter,
                                  int nsource,
                                  ap_uint<32>* sourceID,
                                  buffType* offsetCSC,
                                  buffType* indiceCSC,
                                  buffType* weightCSC,
                                  buffType* degreeCSR,
                                  buffType* cntValFull0,
                                  buffType* buffPing0,
                                  buffType* buffPong0,
                                  buffType* cntValFull1,
                                  buffType* buffPing1,
                                  buffType* buffPong1,
                                  int* resultInfo,
                                  ap_uint<widthOr>* orderUnroll) {
    const int depthOffset = depOffset;
    const int depthVertex = depVertex;
    const int depthEdge = depEdge;
    const int cacheDepthBin =
        (CHANNEL_NUM == 6) ? 12 : 14; // cache line depth in Binary per channel, total (1<<cacheDepthBin)*channels
    const int AxiLatency = (cacheDepthBin == 0) ? 32 : 64;

    const int OffChipLatency = 36; // 64;

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = sourceID latency = \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 32 depth = depthOffset
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = offsetCSC latency = \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 32 depth = depthOffset
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = orderUnroll latency =          \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 32 num_write_outstanding = \
        32 max_write_burst_length = 32 depth = depthOffset

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = indiceCSC latency = \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 32 depth = depthEdge
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem4 port = weightCSC latency = \
    OffChipLatency num_read_outstanding = 32 max_read_burst_length = 32 depth = depthEdge

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem5 port = degreeCSR latency = AxiLatency num_read_outstanding = \
    256 max_read_burst_length = 2 num_write_outstanding = 256 max_write_burst_length = 2 depth = depthOffset
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem6 port = cntValFull0 latency =          \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem7 port = buffPing0 latency =            \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem8 port = buffPong0 latency =            \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem9 port = cntValFull1 latency =          \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem10 port = buffPing1 latency =           \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem11 port = buffPong1 latency =           \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem7 port = resultInfo latency =           \
    OffChipLatency num_read_outstanding = 256 max_read_burst_length = 2 num_write_outstanding = \
        2 max_write_burst_length = 32 depth = 2

#pragma HLS INTERFACE s_axilite port = sourceID bundle = control
#pragma HLS INTERFACE s_axilite port = offsetCSC bundle = control
#pragma HLS INTERFACE s_axilite port = indiceCSC bundle = control
#pragma HLS INTERFACE s_axilite port = weightCSC bundle = control
#pragma HLS INTERFACE s_axilite port = degreeCSR bundle = control

#pragma HLS INTERFACE s_axilite port = cntValFull0 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing0 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong0 bundle = control
#pragma HLS INTERFACE s_axilite port = cntValFull1 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPing1 bundle = control
#pragma HLS INTERFACE s_axilite port = buffPong1 bundle = control

#pragma HLS INTERFACE s_axilite port = resultInfo bundle = control
#pragma HLS INTERFACE s_axilite port = orderUnroll bundle = control

#pragma HLS INTERFACE s_axilite port = nsource bundle = control
#pragma HLS INTERFACE s_axilite port = nnz bundle = control
#pragma HLS INTERFACE s_axilite port = nrows bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = tolerance bundle = control
#pragma HLS INTERFACE s_axilite port = maxIter bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    DT randomProbability = 1.0; // the init of PR, normally use 1.0 or 1.0/nVertex
    const int widthT = (BITS_IN_BYTE * sizeof(DT));
    const int dataOneLineBin = unrollBin;       // data numbers in Binary of one buffType
    const int dataOneLineBin2 = floatUnrollBin; // data numbers in Binary of one buffType
    const int usURAM =
        (cacheDepthBin == 0) ? 0 : 1; // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::pageRankTop<DT, maxVertex, maxEdge, unrollBin, widthOr, cacheDepthBin, dataOneLineBin, dataOneLineBin2,
                           usURAM>(nrows, nnz, nsource, sourceID, degreeCSR, offsetCSC, indiceCSC, weightCSC,
                                   cntValFull0, buffPing0, buffPong0, cntValFull1, buffPing1, buffPong1, orderUnroll,
                                   resultInfo, randomProbability, alpha, tolerance, maxIter);
}
#endif

#endif
