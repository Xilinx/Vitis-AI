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

#ifndef _XF_GRAPH_CONVERT_HPP_
#define _XF_GRAPH_CONVERT_HPP_

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

#define N 1
#define K 16
#define W 32
#define LEN 128

typedef int DT;
typedef ap_uint<512> uint512;

extern "C" void convertCsrCsc_kernel(int vertexNum,
                                     int edgeNum,
                                     uint512* offsetG1,
                                     uint512* indexG1,
                                     uint512* offsetG2,
                                     DT* indexG2,
                                     uint512* offsetG2Tmp1,
                                     uint512* offsetG2Tmp2);

#endif
