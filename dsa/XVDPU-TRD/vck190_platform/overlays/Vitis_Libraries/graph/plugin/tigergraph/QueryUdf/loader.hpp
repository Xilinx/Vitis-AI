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

#include "graph.hpp"
#include <dlfcn.h>
#include <gle/engine/cpplib/headers.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int bfs_fpga_wrapper(int numVertices,
                     int numEdges,
                     int sourceID,
                     xf::graph::Graph<unsigned int, unsigned int> g,
                     unsigned int* predecent,
                     unsigned int* distance);

int shortest_ss_pos_wt_fpga_wrapper(uint32_t numVertices,
                                    uint32_t sourceID,
                                    bool weighted,
                                    xf::graph::Graph<uint32_t, float> g,
                                    float** result,
                                    uint32_t** pred);

int load_xgraph_fpga_wrapper(uint32_t numVertices, uint32_t numEdges, xf::graph::Graph<uint32_t, float> g);

int pageRank_wt_fpga_wrapper(
    float alpha, float tolerance, uint32_t maxIter, xf::graph::Graph<uint32_t, float> g, float* rank);

int load_xgraph_pageRank_wt_fpga_wrapper(uint32_t numVertices, uint32_t numEdges, xf::graph::Graph<uint32_t, float> g);

int load_xgraph_cosine_nbor_ss_fpga_wrapper(uint32_t numVertices,
                                            uint32_t numEdges,
                                            xf::graph::Graph<uint32_t, float> g);

int cosine_nbor_ss_fpga_wrapper(uint32_t topK,
                                uint32_t sourceLen,
                                uint32_t* sourceIndice,
                                uint32_t* sourceWeight,
                                xf::graph::Graph<uint32_t, float> g,
                                uint32_t* resultID,
                                float* similarity);

int loadgraph_cosinesim_ss_dense_fpga_wrapper(uint32_t deviceNeeded,
                                              uint32_t cuNm,
                                              xf::graph::Graph<int32_t, int32_t>** g);

int cosinesim_ss_dense_fpga(uint32_t deviceNeeded,
                            int32_t sourceLen,
                            int32_t* sourceWeight,
                            int32_t topK,
                            xf::graph::Graph<int32_t, int32_t>** g,
                            int32_t* resultID,
                            float* similarity);

int close_fpga();
