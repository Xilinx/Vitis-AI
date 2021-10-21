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

#include "xf_graph_L2.hpp"

#define DT double
#define unrollBin 3 // unroll order
#define unrollNm (1 << unrollBin)
#define maxVertex (67108864 / unrollNm)
#define maxEdge (67108864 / unrollNm)

typedef ap_uint<512> buffType;

extern "C" void kernel_calcuDegree_0(int nrows, int nnz, buffType* degreeCSR, buffType* indiceCSC);
