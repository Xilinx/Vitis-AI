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

#ifndef _XF_GRAPH_SHORTESTPATH_KERNEL_HPP_
#define _XF_GRAPH_SHORTESTPATH_KERNEL_HPP_

#include "xf_graph_L2.hpp"
#include <ap_int.h>
#include <hls_stream.h>

// Vertex number
// Edge number
#ifdef HLS_TEST
#define V 2
#define E 2
#else
#define V 80000000
#define E 80000000
#endif

#define MAXOUTDEGREE (4096 * 10)

extern "C" void shortestPath_top(ap_uint<32>* config,
                                 ap_uint<512>* offset,
                                 ap_uint<512>* column,
                                 ap_uint<512>* weight,

                                 ap_uint<512>* ddrQue512,
                                 ap_uint<32>* ddrQue,

                                 ap_uint<512>* result512,
                                 ap_uint<64>* result,
                                 ap_uint<512>* pred512,
                                 ap_uint<32>* pred,
                                 ap_uint<8>* info);

#endif
