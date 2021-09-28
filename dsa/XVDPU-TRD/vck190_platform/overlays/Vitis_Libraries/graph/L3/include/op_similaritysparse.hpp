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

#ifndef _XF_GRAPH_L3_OP_SIMILARITYSPARSE_HPP_
#define _XF_GRAPH_L3_OP_SIMILARITYSPARSE_HPP_

#include "graph.hpp"
#include "op_base.hpp"
#include "openclHandle.hpp"

namespace xf {
namespace graph {
namespace L3 {

class opSimilaritySparse : public opBase {
   public:
    static uint32_t cuPerBoardSimSparse;

    static uint32_t dupNmSimSparse;

    std::thread simSparseThread;

    std::vector<event<int> > eventQueue;

    class clHandle* handles;

    opSimilaritySparse() : opBase(){};

    void setHWInfo(uint32_t numDev, uint32_t CUmax);

    void freeSimSparse();

    void init(char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad);

    void loadGraph(xf::graph::Graph<uint32_t, float> g); // loadGraph only support loading of CSR format graph

    static int compute(unsigned int deviceID,
                       unsigned int cuID,
                       unsigned int channelID,
                       xrmContext* ctx,
                       xrmCuResource* resR,
                       std::string instanceName,
                       clHandle* handles,
                       uint32_t similarityType,
                       uint32_t dataType,
                       uint32_t sourceNUM,
                       uint32_t* sourceIndice,
                       uint32_t* sourceWeight,
                       uint32_t topK,
                       xf::graph::Graph<uint32_t, float> g,
                       uint32_t* resultID,
                       float* similarity);

    event<int> addwork(uint32_t similarityType,
                       uint32_t dataType,
                       uint32_t sourceNUM,
                       uint32_t* sourceIndice,
                       uint32_t* sourceWeight,
                       uint32_t topK,
                       xf::graph::Graph<uint32_t, float> g,
                       uint32_t* resultID,
                       float* similarity);

    static int computeKNN(unsigned int deviceID,
                          unsigned int cuID,
                          unsigned int channelID,
                          xrmContext* ctx,
                          xrmCuResource* resR,
                          std::string instanceName,
                          clHandle* handles,
                          uint32_t similarityType,
                          uint32_t dataType,
                          uint32_t sourceNUM,
                          uint32_t* sourceIndice,
                          uint32_t* sourceWeight,
                          uint32_t topK,
                          xf::graph::Graph<uint32_t, float> g,
                          std::string* knownLabels,
                          std::string* label);

    event<int> addworkKNN(uint32_t similarityType,
                          uint32_t dataType,
                          uint32_t sourceNUM,
                          uint32_t* sourceIndice,
                          uint32_t* sourceWeight,
                          uint32_t topK,
                          xf::graph::Graph<uint32_t, float> g,
                          std::string* knownLabels,
                          std::string& label);

    static int computeAP(unsigned int deviceID,
                         unsigned int cuID,
                         unsigned int channelID,
                         xrmContext* ctx,
                         xrmCuResource* resR,
                         std::string instanceName,
                         clHandle* handles,
                         uint32_t similarityType,
                         uint32_t dataType,
                         uint32_t sourceID,
                         uint32_t topK,
                         xf::graph::Graph<uint32_t, float> g,
                         uint32_t* resultID,
                         float* similarity);

    event<int> addworkAP(uint32_t similarityType,
                         uint32_t dataType,
                         uint32_t sourceID,
                         uint32_t topK,
                         xf::graph::Graph<uint32_t, float> g,
                         uint32_t* resultID,
                         float* similarity);

    static int computeAPKNN(unsigned int deviceID,
                            unsigned int cuID,
                            unsigned int channelID,
                            xrmContext* ctx,
                            xrmCuResource* resR,
                            std::string instanceName,
                            clHandle* handles,
                            uint32_t similarityType,
                            uint32_t dataType,
                            uint32_t sourceID,
                            uint32_t topK,
                            xf::graph::Graph<uint32_t, float> g,
                            std::string* knownLabels,
                            std::string* label);

    event<int> addworkAPKNN(uint32_t similarityType,
                            uint32_t dataType,
                            uint32_t sourceID,
                            uint32_t topK,
                            xf::graph::Graph<uint32_t, float> g,
                            std::string* knownLabels,
                            std::string& label);

   private:
    std::vector<int> deviceOffset;

    uint32_t deviceNm;

    uint32_t maxCU;

    static void bufferInit(clHandle* hds,
                           std::string instanceName0,
                           xf::graph::Graph<uint32_t, float> g,
                           int similarityType,
                           int dataType,
                           uint32_t topK,
                           unsigned int sourceNUM,
                           uint32_t* sourceIndice,
                           uint32_t* sourceWeight,
                           uint32_t* config,
                           uint32_t* resultID,
                           float* similarity,
                           cl::Kernel& kernel0,
                           std::vector<cl::Memory>& ob_in,
                           std::vector<cl::Memory>& ob_out);

    static int cuExecute(
        clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut);

    static void migrateMemObj(clHandle* hds,
                              bool type,
                              unsigned int num_runs,
                              std::vector<cl::Memory>& ob,
                              std::vector<cl::Event>* evIn,
                              cl::Event* evOut);

    static void cuRelease(xrmContext* ctx, xrmCuResource* resR);

    static void postProcessKNN(
        uint32_t topK, std::string* knownLabels, uint32_t* resultID, float* similarity, std::string* label);
};
} // L3
} // graph
} // xf

#endif
