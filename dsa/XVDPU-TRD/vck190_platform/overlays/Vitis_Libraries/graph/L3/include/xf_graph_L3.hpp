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

#ifndef _XF_GRAPH_L3_HPP_
#define _XF_GRAPH_L3_HPP_

#include "xf_graph_L3_handle.hpp"

namespace xf {
namespace graph {
namespace L3 {

/**
 * @brief twoHop algorithm is implemented.
 *
 * @param handle Graph library L3 handle
 * @param numPart Number of pairs of each part to be counted.
 * @param pairPart Source and destination pairs of each part to be counted.
 * @param resPart result of each part. The order matches the order of the input pairPart.
 * @param g Input, CSR graph of IDs' type of uint32_t and weights' type of float
 *
 */
event<int> twoHop(xf::graph::L3::Handle& handle,
                  uint32_t* numPart,
                  uint64_t** pairPart,
                  uint32_t** resPart,
                  xf::graph::Graph<uint32_t, float> g);

/**
 * @brief pageRank algorithm is implemented.
 *
 * @param handle Graph library L3 handle
 * @param alpha Damping factor, normally 0.85
 * @param tolerance Converge tolerance
 * @param maxIter Max iteration
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param pagerank Output, float type rank values of each vertex
 *
 */
event<int> pageRankWeight(xf::graph::L3::Handle& handle,
                          float alpha,
                          float tolerance,
                          int maxIter,
                          xf::graph::Graph<uint32_t, float> gr,
                          float* pagerank);

/**
 * @brief The single source shortest path algorithm is implemented, the input is the matrix in CSR format.
 *
 * @param handle Graph library L3 handle
 * @param nSource Number of source vertices
 * @param sourceID IDs of giving source vertices
 * @param weighted Bool type flag, when weighted is flag 0, all weights are treated as 1, and when weighted flag is 1,
 * the weights in the gr will be used
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param result The distance result from giving source vertices for each vertex
 * @param predecent The result of parent index of each vertex from giving source vertices for each vertex
 *
 */
event<int> shortestPath(xf::graph::L3::Handle& handle,
                        uint32_t nSource,
                        uint32_t* sourceID,
                        bool weighted,
                        xf::graph::Graph<uint32_t, float> gr,
                        float** result,
                        uint32_t** predecent);

/**
 * @brief The single source cosine similarity API for sparse graph.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceIndices buffer length of source vertex
 * @param sourceIndices Input, source vertex's out members
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
event<int> cosineSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceIndices,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> gr,
                                    uint32_t* resultID,
                                    float* similarity);

/**
 * @brief The single source jaccard similarity API for sparse graph.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceIndices buffer length of source vertex
 * @param sourceIndices Input, source vertex's out members
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
event<int> jaccardSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                     uint32_t sourceNUM,
                                     uint32_t* sourceIndices,
                                     uint32_t* sourceWeights,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> gr,
                                     uint32_t* resultID,
                                     float* similarity);

/**
 * @brief The all-pairs cosine similarity API for sparse graph.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs of all vertices in the sparse graph
 * @param similarity Output, similarity values of all vertices in the sparse graph
 *
 */
event<int> cosineSimilarityAPSparse(xf::graph::L3::Handle& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> gr,
                                    uint32_t** resultID,
                                    float** similarity);

/**
 * @brief The all-pairs jaccard similarity API for sparse graph.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs of all vertices in the sparse graph
 * @param similarity Output, similarity values of all vertices in the sparse graph
 *
 */
event<int> jaccardSimilarityAPSparse(xf::graph::L3::Handle& handle,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> gr,
                                     uint32_t** resultID,
                                     float** similarity);

/**
 * @brief The single source cosine similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceWeights buffer length of source vertex
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
event<int> cosineSimilaritySSDense(xf::graph::L3::Handle& handle,
                                   uint32_t sourceNUM,
                                   uint32_t* sourceWeights,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> gr,
                                   uint32_t* resultID,
                                   float* similarity);

/**
 * @brief The Multi-cards' single source cosine similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param deviceNm FPGA card ID
 * @param sourceNUM Input, sourceWeights buffer length of source vertex
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of int32_t and weights' type of int32_t
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
int cosineSimilaritySSDenseMultiCardBlocking(xf::graph::L3::Handle& handle,
                                             int32_t deviceNm,
                                             int32_t sourceNUM,
                                             int32_t* sourceWeights,
                                             int32_t topK,
                                             xf::graph::Graph<int32_t, int32_t>** gr,
                                             int32_t* resultID,
                                             float* similarity);

/**
 * @brief The Non-blocking Multi-cards' single source cosine similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param deviceNm FPGA card ID
 * @param sourceNUM Input, sourceWeights buffer length of source vertex
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of int32_t and weights' type of int32_t
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
std::vector<event<int> > cosineSimilaritySSDenseMultiCard(xf::graph::L3::Handle& handle,
                                                          int32_t deviceNm,
                                                          int32_t sourceNUM,
                                                          int32_t* sourceWeights,
                                                          int32_t topK,
                                                          xf::graph::Graph<int32_t, int32_t>** g,
                                                          int32_t** resultID,
                                                          float** similarity);

/**
 * @brief The single source jaccard similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceWeights buffer length of source vertex
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs
 * @param similarity Output, similarity values corresponding to theirs IDs
 *
 */
event<int> jaccardSimilaritySSDense(xf::graph::L3::Handle& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> gr,
                                    uint32_t* resultID,
                                    float* similarity);

/**
 * @brief The all-pairs cosine similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs of all vertices in the sparse graph
 * @param similarity Output, similarity values of all vertices in the sparse graph
 *
 */
event<int> cosineSimilarityAPDense(xf::graph::L3::Handle& handle,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> gr,
                                   uint32_t** resultID,
                                   float** similarity);

/**
 * @brief The all-pairs jaccard similarity API for dense graph.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param resultID Output, the topK highest similarity IDs of all vertices in the sparse graph
 * @param similarity Output, similarity values of all vertices in the sparse graph
 *
 */
event<int> jaccardSimilarityAPDense(xf::graph::L3::Handle& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> gr,
                                    uint32_t** resultID,
                                    float** similarity);

/**
 * @brief The single source k-nearest neighbors API for sparse graph. knnSimilarity API is based on the cosine
 * similarity algorithm.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceIndices buffer length of source vertex
 * @param sourceIndices Input, source vertex's out members
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param knownLabels Input, labels of each vertex in the sparse graph
 * @param label Output, the predicted most similar label
 *
 */
event<int> knnSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                 uint32_t sourceNUM,
                                 uint32_t* sourceIndices,
                                 uint32_t* sourceWeights,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> gr,
                                 std::string* knownLabels,
                                 std::string& label);

/**
 * @brief The all-pairs k-nearest neighbors API for sparse graph. knnSimilarity API is based on the cosine similarity
 * algorithm.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param knownLabels Input, labels of each vertex in the sparse graph
 * @param label Output, the predicted most similar labels of all vertices in the sparse graph
 *
 */
event<int> knnSimilarinyAPSparse(xf::graph::L3::Handle& handle,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> gr,
                                 std::string* knownLabels,
                                 std::string* label);

/**
 * @brief The single source k-nearest neighbors API for dense graph. knnSimilarity API is based on the cosine similarity
 * algorithm.
 *
 * @param handle Graph library L3 handle
 * @param sourceNUM Input, sourceIndices buffer length of source vertex
 * @param sourceWeights Input, weights of the source vertex's out members
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param knownLabels Input, labels of each vertex in the dense graph
 * @param label Output, the predicted most similar label
 *
 */
event<int> knnSimilaritySSDense(xf::graph::L3::Handle& handle,
                                uint32_t sourceNUM,
                                uint32_t* sourceWeights,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> gr,
                                std::string* knownLabels,
                                std::string& label);

/**
 * @brief The all-pairs k-nearest neighbors API for dense graph. knnSimilarity API is based on the cosine similarity
 * algorithm.
 *
 * @param handle Graph library L3 handle
 * @param topK Input, the output similarity buffer length
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of float
 * @param knownLabels Input, labels of each vertex in the dense graph
 * @param label Output, the predicted most similar labels of all vertices in the dense graph
 *
 */
event<int> knnSimilarityAPDense(xf::graph::L3::Handle& handle,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> gr,
                                std::string* knownLabels,
                                std::string* label);

/**
 * @brief triangleCount the triangle counting algorithm is implemented, the input is the matrix in CSC format.
 *
 * @param handle Graph library L3 handle
 * @param gr Input, CSR/CSC graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param nTriangle Return triangles number
 *
 */
event<int> triangleCount(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> gr, uint64_t& nTriangle);

/**
 * @brief labelPropagation the label propagation algorithm is implemented
 *
 * @param handle Graph library L3 handle
 * @param maxIter Max iteration
 * @param gr Input, CSR/CSC graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param labels Output labels
 *
 */
event<int> labelPropagation(xf::graph::L3::Handle& handle,
                            uint32_t maxIter,
                            xf::graph::Graph<uint32_t, uint32_t> gr,
                            uint32_t* labels);

/**
 * @brief bfs Implements the directed graph traversal by breath-first search algorithm
 *
 * @param handle Graph library L3 handle
 * @param sourceID The source vertex ID in this search, starting from 0
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param predecent The result of parent index of each vertex
 * @param distance The distance result from giving source vertex for each vertex
 *
 */
event<int> bfs(xf::graph::L3::Handle& handle,
               uint32_t sourceID,
               xf::graph::Graph<uint32_t, uint32_t> gr,
               uint32_t* predecent,
               uint32_t* distance);

/**
 * @brief connectedComponents Computes the connected component membership of each vertex only for undirected graph
 *
 * @param handle Graph library L3 handle
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param result Ouput, result buffer with the vertex label containing the lowest vertex id in the strongly
 * connnected component containing that vertex
 *
 */
event<int> wcc(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> gr, uint32_t* result);

/**
 * @brief stronglyConnectedComponents Computes the strongly connected component membership of each vertex only for
 * directed graph, and label each vertex with one value containing the lowest vertex id in the SCC containing
 * that vertex.
 *
 * @param handle Graph library L3 handle
 * @param gr Input, CSR graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param result Ouput, result buffer with the vertex label containing the lowest vertex id in the strongly
 * connnected component containing that vertex
 *
 */
event<int> scc(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> gr, uint32_t* result);

/**
 * @brief Convert from CSR to CSC, now it supports only CSR to CSC of unweighted graph
 *
 * @param handle Graph library L3 handle
 * @param gr1 Input, CSR/CSC graph of IDs' type of uint32_t and weights' type of uint32_t
 * @param gr2 Output, CSC/CSR graph of IDs' type of uint32_t and weights' type of uint32_t
 *
 */
event<int> convertCsrCsc(xf::graph::L3::Handle& handle,
                         xf::graph::Graph<uint32_t, uint32_t> gr1,
                         xf::graph::Graph<uint32_t, uint32_t> gr2);
} // L3
} // graph
} // xf
#endif
