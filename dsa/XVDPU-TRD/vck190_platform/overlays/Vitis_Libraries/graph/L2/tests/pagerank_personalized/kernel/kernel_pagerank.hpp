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

#include "xf_graph_L2.hpp"

// for Calculation accuracy of PageRank
// typedef double DT;
typedef float DT;

#define BITS_IN_BYTE (8)
#define SIZE_OF_DOUBLE (8)
#define doubleUnrollBin (3)
#define floatUnrollBin (4)
#define doubleUnrollNm (1 << doubleUnrollBin)
#define floatUnrollNm (1 << floatUnrollBin)
#define unrollBin ((sizeof(DT) == SIZE_OF_DOUBLE) ? doubleUnrollBin : floatUnrollBin)
#define unrollNm (1 << unrollBin)
#define widthOr ((sizeof(DT) == SIZE_OF_DOUBLE) ? 256 : 512)
#define SIZE_OF_VERTEX (1 << 26)
#define SIZE_OF_EDGE (1 << 27)
#define maxVertex (SIZE_OF_VERTEX / unrollNm)
#define maxEdge (SIZE_OF_EDGE / unrollNm)

typedef ap_uint<512> buffType;

// just for cosim
#define testVertex 40004
#define testEdge 40004
#define depOffset ((testVertex + floatUnrollNm - 1) / floatUnrollNm) // offsetCSC, degreeCSR buffer depth of 512 bits
#define depVertex \
    ((testVertex + unrollNm - 1) / unrollNm) // pagerank, cntValFull, buffPing, buffPong buffers depth of 512 bits
#define depEdge ((testEdge + unrollNm - 1) / unrollNm) // indiceCSC buffer depth of 512 bits

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
                                  ap_uint<widthOr>* orderUnroll);
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
                                  ap_uint<widthOr>* orderUnroll);
#endif

/*
 * widthOr is unrollNm*sizeof(int) for the read speed matching
 * maxVertex is limited by 64M in the design
 * maxEdge is limited by 128M in the design
 * The max offsetCSC speaces will be 64M*4bits      256MB
 * The max indiceCSC speaces will be 128M*4bits     512MB
 * The max weightCSC speaces will be 128M*4bits     512MB
 * The max degreeCSR speaces will be 64M*4bits      256MB
 * The max constVal speaces will be 64M*sizeof(DT)  512MB
 * The max buffPing speaces will be 64M*sizeof(DT)  512MB
 * The max buffPong speaces will be 64M*sizeof(DT)  512MB
 * The max orderUnroll speaces will be 64M*4bits    256MB
 */
