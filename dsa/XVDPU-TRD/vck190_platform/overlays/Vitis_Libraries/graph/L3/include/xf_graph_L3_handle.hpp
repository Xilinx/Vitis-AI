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

#ifndef _XF_GRAPH_L3_HANDLE_HPP_
#define _XF_GRAPH_L3_HANDLE_HPP_

#include "op_pagerank.hpp"
#include "op_sp.hpp"
#include "op_trianglecount.hpp"
#include "op_labelpropagation.hpp"
#include "op_bfs.hpp"
#include "op_wcc.hpp"
#include "op_scc.hpp"
#include "op_convertcsrcsc.hpp"
#include "op_similaritysparse.hpp"
#include "op_similaritydense.hpp"
#include "op_twohop.hpp"

namespace xf {
namespace graph {
namespace L3 {
/**
 * @brief Graph library L3 handle
 *
 */
class Handle {
   public:
    /**
     * \brief setup information of single operation
     *
     * \param operationName operation's name
     * \param kernelName kernel's name, if kernelName is defined, kernelAlias will not be needed
     * \param kernelAlias kernel's name, if kernelName is not defined, kernelName will not be needed
     * \param requestLoad percentation of computing unit's occupation. Maximum is 100, which represents that 100% of the
     * computing unit are occupied. By using lower requestLoad, the hardware resources can be pipelined and higher
     * throughput can be achieved
     * \param xclbinFile xclbin file path
     * \param deviceNeeded needed FPGA board number
     * \param deviceIDs FPGA board IDs
     *
     */
    struct singleOP {
        char* operationName; // for example, cosineSim
        char* kernelName;    // user defined kernel names
        char* kernelAlias;   // user defined kernel names
        unsigned int requestLoad = 100;
        char* xclbinFile;                    // xclbin full path
        char* xclbinFile2;                   // xclbin full path
        unsigned int deviceNeeded = 0;       // requested FPGA device number
        unsigned int cuPerBoard = 1;         // requested FPGA device number
        std::vector<unsigned int> deviceIDs; // deviceID
        void setKernelName(char* input) {
            std::string tmp = "";
            kernelName = input;
            kernelAlias = (char*)tmp.c_str();
        }
        void setKernelAlias(char* input) {
            std::string tmp = "";
            kernelName = (char*)tmp.c_str();
            kernelAlias = input;
        }
    };

    /**
     * \brief twohop operation
     *
     */
    class opTwoHop* optwohop;
    /**
     * \brief pageRank operation
     *
     */
    class opPageRank* oppg;
    /**
     * \brief shortest path operation
     *
     */
    class opSP* opsp;
    /**
     * \brief triangle count operation
     *
     */
    class opTriangleCount* optcount;
    /**
     * \brief label propagation operation
     *
     */
    class opLabelPropagation* oplprop;
    /**
     * \brief breadth-first search operation
     *
     */
    class opBFS* opbfs;
    /**
     * \brief weakly connected components operation
     *
     */
    class opWCC* opwcc;
    /**
     * \brief strongly connected components operation
     *
     */
    class opSCC* opscc;
    /**
     * \brief convert CSR/CSC operation
     *
     */
    class opConvertCsrCsc* opconvertcsrcsc;
    /**
     * \brief sparse matrix cosine/jaccard similarity operation
     *
     */
    class opSimilaritySparse* opsimsparse;
    /**
     * \brief dense matrix cosine/jaccard similarity operation
     *
     */
    class opSimilarityDense* opsimdense;
    /**
     * \brief xilinx FPGA Resource Manager operation
     *
     */
    class openXRM* xrm;

    Handle() {
        optwohop = new class opTwoHop;
        oppg = new class opPageRank;
        opsp = new class opSP;
        optcount = new class opTriangleCount;
        oplprop = new class opLabelPropagation;
        opbfs = new class opBFS;
        opwcc = new class opWCC;
        opscc = new class opSCC;
        opconvertcsrcsc = new class opConvertCsrCsc;
        opsimsparse = new class opSimilaritySparse;
        opsimdense = new class opSimilarityDense;
        xrm = new class openXRM;
    };

    void free();

    int setUp();

    void getEnv();

    void addOp(singleOP op);

   private:
    uint32_t maxCU;

    uint32_t deviceNm;

    uint32_t numDevices;

    uint64_t maxChannelSize;

    std::vector<singleOP> ops;

    void loadXclbin(unsigned int deviceId, char* xclbinName);

    std::thread loadXclbinNonBlock(unsigned int deviceId, char* xclbinName);
    std::future<int> loadXclbinAsync(unsigned int deviceId, char* xclbinName);

    void initOpTwoHop(const char* kernelName,
                      char* xclbinFile,
                      char* kernelAlias,
                      unsigned int requestLoad,
                      unsigned int deviceNeeded,
                      unsigned int cuPerBoard);

    void initOpPageRank(const char* kernelName,
                        char* xclbinFile,
                        char* kernelAlias,
                        unsigned int requestLoad,
                        unsigned int deviceNeeded,
                        unsigned int cuPerBoard);

    void initOpSP(const char* kernelName,
                  char* xclbinFile,
                  char* kernelAlias,
                  unsigned int requestLoad,
                  unsigned int deviceNeeded,
                  unsigned int cuPerBoard);

    void initOpTriangleCount(const char* kernelName,
                             char* xclbinFile,
                             char* kernelAlias,
                             unsigned int requestLoad,
                             unsigned int deviceNeeded,
                             unsigned int cuPerBoard);

    void initOpLabelPropagation(const char* kernelName,
                                char* xclbinFile,
                                char* kernelAlias,
                                unsigned int requestLoad,
                                unsigned int deviceNeeded,
                                unsigned int cuPerBoard);

    void initOpBFS(const char* kernelName,
                   char* xclbinFile,
                   char* kernelAlias,
                   unsigned int requestLoad,
                   unsigned int deviceNeeded,
                   unsigned int cuPerBoard);

    void initOpWCC(const char* kernelName,
                   char* xclbinFile,
                   char* kernelAlias,
                   unsigned int requestLoad,
                   unsigned int deviceNeeded,
                   unsigned int cuPerBoard);

    void initOpSCC(const char* kernelName,
                   char* xclbinFile,
                   char* kernelAlias,
                   unsigned int requestLoad,
                   unsigned int deviceNeeded,
                   unsigned int cuPerBoard);

    void initOpConvertCsrCsc(const char* kernelName,
                             char* xclbinFile,
                             char* kernelAlias,
                             unsigned int requestLoad,
                             unsigned int deviceNeeded,
                             unsigned int cuPerBoard);

    void initOpSimSparse(const char* kernelName,
                         char* xclbinFile,
                         char* kernelAlias,
                         unsigned int requestLoad,
                         unsigned int deviceNeeded,
                         unsigned int cuPerBoard);

    void initOpSimDense(const char* kernelName,
                        char* xclbinFile,
                        char* kernelAlias,
                        unsigned int requestLoad,
                        unsigned int deviceNeeded,
                        unsigned int cuPerBoard);

    void initOpSimDenseInt(const char* kernelName,
                           char* xclbinFile,
                           char* xclbinFile2,
                           char* kernelAlias,
                           unsigned int requestLoad,
                           unsigned int deviceNeeded,
                           unsigned int cuPerBoard);
};
} // L3
} // graph
} // xf
#endif
