/*
 * Copyright 2020 Xilinx, Inc.
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
 * @file dut.h
 *
 * @brief This file contains top function of test case.
 */

#include "louvain_modularity.hpp"

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#define DWIDTHS (256)
#define CSRWIDTHS (256)
#define COLORWIDTHS (32)
#define NUM (DWIDTHS / 32)
#define MAXNV (1 << 26)
#define MAXNE (1 << 27)
#define VERTEXS (MAXNV / NUM)
#define EDGES (MAXNE / NUM)
#define DEGREES (1 << 17)
#define COLORS (4096)

const int depthVertex = VERTEXS;
const int depthEdge = EDGES;

typedef double DWEIGHT;

extern "C" void kernel_louvain(int64_t* config0,
                               DWEIGHT* config1,
                               ap_uint<CSRWIDTHS>* offsets,
                               ap_uint<CSRWIDTHS>* indices,
                               ap_uint<CSRWIDTHS>* weights,
                               ap_uint<COLORWIDTHS>* colorAxi,
                               ap_uint<COLORWIDTHS>* colorInx,
                               ap_uint<DWIDTHS>* cidPrev,
                               ap_uint<DWIDTHS>* cidSizePrev,
                               ap_uint<DWIDTHS>* totPrev,
                               ap_uint<DWIDTHS>* cidCurr,
                               ap_uint<DWIDTHS>* cidSizeCurr,
                               ap_uint<DWIDTHS>* totCurr,
                               ap_uint<DWIDTHS>* cidSizeUpdate,
                               ap_uint<DWIDTHS>* totUpdate,
                               ap_uint<DWIDTHS>* cWeight,
                               ap_uint<CSRWIDTHS>* offsetsDup,
                               ap_uint<CSRWIDTHS>* indicesDup,
                               ap_uint<8>* flag,
                               ap_uint<8>* flagUpdate);
