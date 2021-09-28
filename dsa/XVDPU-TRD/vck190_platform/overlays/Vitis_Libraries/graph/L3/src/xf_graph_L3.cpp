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

#ifndef _XF_GRAPH_L3_CPP_
#define _XF_GRAPH_L3_CPP_

#include "xf_graph_L3.hpp"

namespace xf {
namespace graph {
namespace L3 {

int runMultiEvents(uint32_t number, std::vector<xf::graph::L3::event<int> >& f) {
    int ret = 0;
    for (int i = 0; i < number; ++i) {
        ret += f[i].wait();
    }
    return ret;
}

event<int> twoHop(xf::graph::L3::Handle& handle,
                  uint32_t* numPart,
                  uint64_t** pairPart,
                  uint32_t** resPart,
                  xf::graph::Graph<uint32_t, float> g) {
    for (int i = 0; i < 5; ++i) {
        (handle.optwohop)
            ->eventQueue.push_back((handle.optwohop)->addwork(numPart[i], &pairPart[i][0], &resPart[i][0], g));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.optwohop)->twoHopThread = std::thread(std::move(t), 5, std::ref((handle.optwohop)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> pageRankWeight(xf::graph::L3::Handle& handle,
                          float alpha,
                          float tolerance,
                          int maxIter,
                          xf::graph::Graph<uint32_t, float> g,
                          float* pagerank) {
    return (handle.oppg)->addwork(alpha, tolerance, maxIter, g, pagerank);
};

event<int> shortestPath(xf::graph::L3::Handle& handle,
                        uint32_t nSource,
                        uint32_t* sourceID,
                        bool weighted,
                        xf::graph::Graph<uint32_t, float> g,
                        float** result,
                        uint32_t** pred) {
    for (int i = 0; i < nSource; ++i) {
        (handle.opsp)
            ->eventQueue.push_back((handle.opsp)->addwork(1, &sourceID[i], weighted, g, &result[i][0], &pred[i][0]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsp)->msspThread = std::thread(std::move(t), nSource, std::ref((handle.opsp)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> triangleCount(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> g, uint64_t& nTriangle) {
    return (handle.optcount)->addwork(g, nTriangle);
};

event<int> labelPropagation(xf::graph::L3::Handle& handle,
                            uint32_t maxIter,
                            xf::graph::Graph<uint32_t, uint32_t> g,
                            uint32_t* labels) {
    return (handle.oplprop)->addwork(maxIter, g, labels);
};

event<int> bfs(xf::graph::L3::Handle& handle,
               uint32_t sourceID,
               xf::graph::Graph<uint32_t, uint32_t> g,
               uint32_t* predecent,
               uint32_t* distance) {
    return (handle.opbfs)->addwork(sourceID, g, predecent, distance);
};

event<int> wcc(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> g, uint32_t* result) {
    return (handle.opwcc)->addwork(g, result);
};

event<int> scc(xf::graph::L3::Handle& handle, xf::graph::Graph<uint32_t, uint32_t> g, uint32_t* result) {
    return (handle.opscc)->addwork(g, result);
};

event<int> convertCsrCsc(xf::graph::L3::Handle& handle,
                         xf::graph::Graph<uint32_t, uint32_t> g,
                         xf::graph::Graph<uint32_t, uint32_t> g2) {
    return (handle.opconvertcsrcsc)->addwork(g, g2);
};

event<int> cosineSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceIndice,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t* resultID,
                                    float* similarity) {
    (handle.opsimsparse)
        ->eventQueue.push_back(
            (handle.opsimsparse)->addwork(1, 1, sourceNUM, sourceIndice, sourceWeights, topK, g, resultID, similarity));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread = std::thread(std::move(t), 1, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> cosineSimilarityAPSparse(xf::graph::L3::Handle& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t** resultID,
                                    float** similarity) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimsparse)
            ->eventQueue.push_back(
                (handle.opsimsparse)->addworkAP(1, 1, i, topK, g, &resultID[i][0], &similarity[i][0]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> jaccardSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                     uint32_t sourceNUM,
                                     uint32_t* sourceIndice,
                                     uint32_t* sourceWeights,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> g,
                                     uint32_t* resultID,
                                     float* similarity) {
    (handle.opsimsparse)
        ->eventQueue.push_back(
            (handle.opsimsparse)->addwork(0, 1, sourceNUM, sourceIndice, sourceWeights, topK, g, resultID, similarity));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread = std::thread(std::move(t), 1, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> jaccardSimilarityAPSparse(xf::graph::L3::Handle& handle,
                                     uint32_t topK,
                                     xf::graph::Graph<uint32_t, float> g,
                                     uint32_t** resultID,
                                     float** similarity) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimsparse)
            ->eventQueue.push_back(
                (handle.opsimsparse)->addworkAP(0, 1, i, topK, g, &resultID[i][0], &similarity[i][0]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> cosineSimilaritySSDense(xf::graph::L3::Handle& handle,
                                   uint32_t sourceNUM,
                                   uint32_t* sourceWeights,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> g,
                                   uint32_t* resultID,
                                   float* similarity) {
    (handle.opsimdense)
        ->eventQueue.push_back(
            (handle.opsimdense)->addwork(1, 1, sourceNUM, sourceWeights, topK, g, resultID, similarity));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread = std::thread(std::move(t), 1, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

std::vector<event<int> > cosineSimilaritySSDenseMultiCard(xf::graph::L3::Handle& handle,
                                                          int32_t deviceNm,
                                                          int32_t sourceNUM,
                                                          int32_t* sourceWeights,
                                                          int32_t topK,
                                                          xf::graph::Graph<int32_t, int32_t>** g,
                                                          int32_t** resultID,
                                                          float** similarity) {
    std::vector<event<int> > eventQueue;
    for (int i = 0; i < deviceNm; ++i) {
        eventQueue.push_back(
            (handle.opsimdense)->addworkInt(1, 0, sourceNUM, sourceWeights, topK, g[i][0], resultID[i], similarity[i]));
    }
    return eventQueue;
};

int cosineSimilaritySSDenseMultiCardBlocking(xf::graph::L3::Handle& handle,
                                             int32_t deviceNm,
                                             int32_t sourceNUM,
                                             int32_t* sourceWeights,
                                             int32_t topK,
                                             xf::graph::Graph<int32_t, int32_t>** g,
                                             int32_t* resultID,
                                             float* similarity) {
    std::vector<event<int> > eventQueue;
    float** similarity0 = new float*[deviceNm];
    int32_t** resultID0 = new int32_t*[deviceNm];
    int counter[deviceNm];
    for (int i = 0; i < deviceNm; ++i) {
        counter[i] = 0;
        similarity0[i] = aligned_alloc<float>(topK);
        resultID0[i] = aligned_alloc<int32_t>(topK);
        memset(resultID0[i], 0, topK * sizeof(int32_t));
        memset(similarity0[i], 0, topK * sizeof(float));
    }
    for (int i = 0; i < deviceNm; ++i) {
        eventQueue.push_back(
            (handle.opsimdense)
                ->addworkInt(1, 0, sourceNUM, sourceWeights, topK, g[i][0], resultID0[i], similarity0[i]));
    }
    int ret = 0;
    for (int i = 0; i < eventQueue.size(); ++i) {
        ret += eventQueue[i].wait();
    }
    for (int i = 0; i < topK; ++i) {
        similarity[i] = similarity0[0][counter[0]];
        int32_t prev = 0;
        resultID[i] = resultID0[0][counter[0]];
        counter[0]++;
        for (int j = 1; j < deviceNm; ++j) {
            if (similarity[i] < similarity0[j][counter[j]]) {
                similarity[i] = similarity0[j][counter[j]];
                resultID[i] = resultID0[j][counter[j]];
                counter[prev]--;
                prev = j;
                counter[j]++;
            }
        }
    }
    for (int i = 0; i < deviceNm; ++i) {
        free(similarity0[i]);
        free(resultID0[i]);
    }
    delete[] similarity0;
    delete[] resultID0;
    return ret;
};

event<int> jaccardSimilaritySSDense(xf::graph::L3::Handle& handle,
                                    uint32_t sourceNUM,
                                    uint32_t* sourceWeights,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t* resultID,
                                    float* similarity) {
    (handle.opsimdense)
        ->eventQueue.push_back(
            (handle.opsimdense)->addwork(0, 1, sourceNUM, sourceWeights, topK, g, resultID, similarity));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread = std::thread(std::move(t), 1, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> cosineSimilarityAPDense(xf::graph::L3::Handle& handle,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> g,
                                   uint32_t** resultID,
                                   float** similarity) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimdense)
            ->eventQueue.push_back(
                (handle.opsimdense)->addworkAP(1, 1, i, topK, g, &resultID[i][0], &similarity[i][0]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> jaccardSimilarityAPDense(xf::graph::L3::Handle& handle,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    uint32_t** resultID,
                                    float** similarity) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimdense)
            ->eventQueue.push_back(
                (handle.opsimdense)->addworkAP(0, 1, i, topK, g, &resultID[i][0], &similarity[i][0]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> knnSimilaritySSSparse(xf::graph::L3::Handle& handle,
                                 uint32_t sourceNUM,
                                 uint32_t* sourceIndice,
                                 uint32_t* sourceWeights,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> g,
                                 std::string* knownLabels,
                                 std::string& label) {
    (handle.opsimsparse)
        ->eventQueue.push_back(
            (handle.opsimsparse)
                ->addworkKNN(1, 1, sourceNUM, sourceIndice, sourceWeights, topK, g, knownLabels, label));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread = std::thread(std::move(t), 1, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> knnSimilaritySSDense(xf::graph::L3::Handle& handle,
                                uint32_t sourceNUM,
                                uint32_t* sourceWeights,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> g,
                                std::string* knownLabels,
                                std::string& label) {
    (handle.opsimdense)
        ->eventQueue.push_back(
            (handle.opsimdense)->addworkKNN(1, 1, sourceNUM, sourceWeights, topK, g, knownLabels, label));
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread = std::thread(std::move(t), 1, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> knnSimilarityAPSparse(xf::graph::L3::Handle& handle,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> g,
                                 std::string* knownLabels,
                                 std::string* label) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimsparse)
            ->eventQueue.push_back((handle.opsimsparse)->addworkAPKNN(1, 1, i, topK, g, knownLabels, label[i]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimsparse)->simSparseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimsparse)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

event<int> knnSimilarityAPDense(xf::graph::L3::Handle& handle,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> g,
                                std::string* knownLabels,
                                std::string* label) {
    uint32_t numVertices = g.nodeNum;
    for (int i = 0; i < numVertices; ++i) {
        (handle.opsimdense)
            ->eventQueue.push_back((handle.opsimdense)->addworkAPKNN(1, 1, i, topK, g, knownLabels, label[i]));
    }
    std::packaged_task<int(uint32_t, std::vector<event<int> >&)> t(runMultiEvents);
    std::future<int> f0 = t.get_future();
    (handle.opsimdense)->simDenseThread =
        std::thread(std::move(t), numVertices, std::ref((handle.opsimdense)->eventQueue));
    return event<int>(std::forward<std::future<int> >(f0));
};

} // L3
} // graph
} // xf
#endif
