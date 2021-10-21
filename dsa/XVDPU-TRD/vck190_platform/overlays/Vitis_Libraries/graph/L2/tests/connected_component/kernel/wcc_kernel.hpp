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

#ifndef _XF_GRAPH_WCC_KERNEL_HPP_
#define _XF_GRAPH_WCC_KERNEL_HPP_

#include "xf_graph_L2.hpp"

#include <ap_int.h>
#include <hls_stream.h>

#ifdef HLS_TEST
// Vertex number
#define V 80
// Edge number
#define E 80
#else
// Vertex number
#define V 80000000
// Edge number
#define E 80000000
#endif

#define N 1
#define K 16
#define W 32
#define LEN 128

// Maximum degree
#define MAXDEGREE (32 * 4096)

typedef ap_uint<32> DT;
typedef ap_uint<512> uint512;

extern "C" void wcc_kernel(const int edgeNum,
                           const int vertexNum,

                           ap_uint<512>* columnG1,
                           ap_uint<512>* offsetG1,
                           ap_uint<512>* column512G2,
                           ap_uint<32>* column32G2,
                           ap_uint<512>* offsetG2,

                           ap_uint<512>* offsetG2Tmp1,
                           ap_uint<512>* offsetG2Tmp2,

                           ap_uint<512>* queue512,
                           ap_uint<32>* queue,

                           ap_uint<512>* result512,
                           ap_uint<32>* result32);

#endif
