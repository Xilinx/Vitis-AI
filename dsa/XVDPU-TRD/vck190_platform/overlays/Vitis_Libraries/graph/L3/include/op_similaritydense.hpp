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

#ifndef _XF_GRAPH_L3_OP_SIMILARITYDENSE_HPP_
#define _XF_GRAPH_L3_OP_SIMILARITYDENSE_HPP_

#include "graph.hpp"
#include "op_base.hpp"
#include "openclHandle.hpp"

namespace xf {
namespace graph {
namespace L3 {

class opSimilarityDense : public opBase {
   public:
    static uint32_t cuPerBoardSimDense;

    static uint32_t dupNmSimDense;

    std::thread simDenseThread;

    std::vector<event<int> > eventQueue;

    class clHandle* handles;

    opSimilarityDense() : opBase(){};

    void setHWInfo(uint32_t numDev, uint32_t CUmax);

    void freeSimDense();

    void init(char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad);

    void initInt(char* kernelName,
                 char* xclbinFile,
                 char* xclbinFile2,
                 uint32_t* deviceIDs,
                 uint32_t* cuIDs,
                 unsigned int requestLoad);

    void loadGraph(xf::graph::Graph<uint32_t, float> g); // loadGraph only support loading of CSR format graph

    void loadGraphMultiCardNonBlocking(int deviceID, int cuID, xf::graph::Graph<int32_t, int32_t> g);

    void loadGraphMultiCardBlocking(int deviceID, int cuID, xf::graph::Graph<int32_t, int32_t> g);

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
                       uint32_t* sourceWeight,
                       uint32_t topK,
                       xf::graph::Graph<uint32_t, float> g,
                       uint32_t* resultID,
                       float* similarity);

    static int computeInt(unsigned int deviceID,
                          unsigned int cuID,
                          unsigned int channelID,
                          xrmContext* ctx,
                          xrmCuResource* resR,
                          std::string instanceName,
                          clHandle* handles,
                          int32_t similarityType,
                          int32_t dataType,
                          int32_t sourceNUM,
                          int32_t* sourceWeight,
                          int32_t topK,
                          xf::graph::Graph<int32_t, int32_t> g,
                          int32_t* resultID,
                          float* similarity);

    event<int> addwork(uint32_t similarityType,
                       uint32_t dataType,
                       uint32_t sourceNUM,
                       uint32_t* sourceWeight,
                       uint32_t topK,
                       xf::graph::Graph<uint32_t, float> g,
                       uint32_t* resultID,
                       float* similarity);

    event<int> addworkInt(int32_t similarityType,
                          int32_t dataType,
                          int32_t sourceNUM,
                          int32_t* sourceWeight,
                          int32_t topK,
                          xf::graph::Graph<int32_t, int32_t> g,
                          int32_t* resultID,
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
                          uint32_t* sourceWeight,
                          uint32_t topK,
                          xf::graph::Graph<uint32_t, float> g,
                          std::string* knownLabels,
                          std::string* label);

    event<int> addworkKNN(uint32_t similarityType,
                          uint32_t dataType,
                          uint32_t sourceNUM,
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
                           uint32_t* sourceWeight,
                           uint32_t* config,
                           uint32_t* resultID,
                           float* similarity,
                           cl::Kernel& kernel0,
                           std::vector<cl::Memory>& ob_in,
                           std::vector<cl::Memory>& ob_out);

    static void bufferInitInt(clHandle* hds,
                              std::string instanceName0,
                              xf::graph::Graph<int32_t, int32_t> g,
                              int cuID,
                              int similarityType,
                              int dataType,
                              int32_t topK,
                              int sourceNUM,
                              int32_t* sourceWeight,
                              uint32_t* config,
                              int32_t* resultID,
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
