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

#pragma once

#ifndef _L3_WRAPPER_HPP_
#define _L3_WRAPPER_HPP_

#include "xf_graph_L3_handle.hpp"
#include <iostream>
#include <memory>
#include <unordered_map>

namespace xf {
namespace graph {
namespace L3 {

event<int> pageRankWeight(std::shared_ptr<xf::graph::L3::Handle>& handle,
                          float alpha,
                          float tolerance,
                          int maxIter,
                          xf::graph::Graph<uint32_t, float> g,
                          float* pagerank);

event<int> shortestPath(std::shared_ptr<xf::graph::L3::Handle>& handle,
                        uint32_t nSource,
                        uint32_t* sourceID,
                        bool weighted,
                        xf::graph::Graph<uint32_t, float> g,
                        float** result,
                        uint32_t** pred);

event<int> cosineSimilaritySSSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceIndice,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t* resultID,
                                    float* similarity);

event<int> jaccardSimilaritySSSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                     uint32_t sourceNUM,
                                     uint32_t* sourceIndice,
                                     uint32_t* sourceWeights,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> g,
                                     uint32_t* resultID,
                                     float* similarity);

event<int> cosineSimilarityAPSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t** resultID,
                                    float** similarity);

event<int> jaccardSimilarityAPSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> g,
                                     uint32_t** resultID,
                                     float** similarity);

event<int> cosineSimilaritySSDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                   uint32_t sourceNUM,
                                   uint32_t* sourceWeights,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> g,
                                   uint32_t* resultID,
                                   float* similarity);

event<int> jaccardSimilaritySSDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t* resultID,
                                    float* similarity);
event<int> cosineSimilarityAPDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> g,
                                   uint32_t** resultID,
                                   float** similarity);
event<int> jaccardSimilarityAPDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t** resultID,
                                    float** similarity);

event<int> knnSimilaritySSSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                 uint32_t sourceNUM,
                                 uint32_t* sourceIndice,
                                 uint32_t* sourceWeights,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> g,
                                 std::string* knownLabels,
                                 std::string& label);

event<int> knnSimilarityAPSparse(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> g,
                                 std::string* knownLabels,
                                 std::string* label);

event<int> knnSimilaritySSDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                uint32_t sourceNUM,
                                uint32_t* sourceWeights,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> g,
                                std::string* knownLabels,
                                std::string& label);

event<int> knnSimilarityAPDense(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> g,
                                std::string* knownLabels,
                                std::string* label);

event<int> triangleCount(std::shared_ptr<xf::graph::L3::Handle>& handle,
                         xf::graph::Graph<uint32_t, uint32_t> g,
                         uint64_t& nTriangle);

event<int> labelPropagation(std::shared_ptr<xf::graph::L3::Handle>& handle,
                            uint32_t maxIter,
                            xf::graph::Graph<uint32_t, uint32_t> g,
                            uint32_t* labels);

event<int> bfs(std::shared_ptr<xf::graph::L3::Handle>& handle,
               uint32_t sourceID,
               xf::graph::Graph<uint32_t, uint32_t> g,
               uint32_t* predecent,
               uint32_t* distance);

event<int> wcc(std::shared_ptr<xf::graph::L3::Handle>& handle,
               xf::graph::Graph<uint32_t, uint32_t> g,
               uint32_t* result);

event<int> scc(std::shared_ptr<xf::graph::L3::Handle>& handle,
               xf::graph::Graph<uint32_t, uint32_t> g,
               uint32_t* result);

/**
 * @brief Convert from CSR to CSC, now it supports only CSR to CSC of unweighted graph
 *
 * @param g graph CSR
 * @param g2 graph CSC
 * @param handle L3 handle
*/
event<int> convertCsrCsc(std::shared_ptr<xf::graph::L3::Handle>& handle,
                         xf::graph::Graph<uint32_t, uint32_t> g,
                         xf::graph::Graph<uint32_t, uint32_t> g2);

int cosineSimilaritySSDenseMultiCard(std::shared_ptr<xf::graph::L3::Handle>& handle,
                                     int32_t deviceNm,
                                     int32_t sourceNUM,
                                     int32_t* sourceWeights,
                                     int32_t topK,
                                     xf::graph::Graph<int32_t, int32_t>** g,
                                     int32_t* resultID,
                                     float* similarity);
int loadGraphMultiCard(std::shared_ptr<xf::graph::L3::Handle>& handle,
                       int32_t deviceNm,
                       int32_t cuNm,
                       xf::graph::Graph<int32_t, int32_t>** g);
std::vector<event<int> > cosineSimilaritySSDenseMultiCard(xf::graph::L3::Handle& handle,
                                                          int32_t deviceNm,
                                                          int32_t sourceNUM,
                                                          int32_t* sourceWeights,
                                                          int32_t topK,
                                                          xf::graph::Graph<int32_t, int32_t>** g,
                                                          int32_t** resultID,
                                                          float** similarity);
} // L3
} // graph
} // xf
#endif
