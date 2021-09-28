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

#ifndef _XF_GRAPH_LABEL_PROPAGATION_TOP_HPP_
#define _XF_GRAPH_LABEL_PROPAGATION_TOP_HPP_

#include "xf_graph_L2.hpp"
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

// Set according to actual needs
// Vertex number
// Edge number
#ifdef HLS_TEST
#define V 2
#define E 2
#else
#define V 800000
#define E 800000
#endif

#define K 16
#define W 32
#define LEN 128

typedef ap_uint<32> DT;
typedef ap_uint<512> uint512;

extern "C" void LPKernel(int vertexNum,
                         int edgeNum,
                         int iterNum,
                         uint512* offsetCSR,
                         uint512* indexCSR,
                         uint512* offsetCSC,
                         uint512* indexCSC,
                         DT* indexCSC2,
                         uint512* pingHashBuf,
                         uint512* pongHashBuf,
                         uint512* labelPing,
                         uint512* labelPong);
#endif
