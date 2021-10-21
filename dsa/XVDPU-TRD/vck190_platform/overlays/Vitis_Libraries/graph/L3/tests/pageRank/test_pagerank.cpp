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

#include <iostream>
#include <string.h>
#include "utils.hpp"
#include "xf_graph_L3.hpp"
#include <cmath>
#include "xf_utils_sw/logger.hpp"

typedef float DT;

// Arguments parser
class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

int main(int argc, const char* argv[]) {
    // Initialize parserl
    ArgParser parser(argc, argv);

    // Initialize paths addresses
    std::string num_str;
    std::string files;
    std::string dataSetDir;
    std::string refDir;
    if (!parser.getCmdOption("-files", num_str)) {
        files = "";
        std::cout << "INFO: dataSet name is not set!\n";
    } else {
        files = num_str;
    }
    if (!parser.getCmdOption("-dataSetDir", num_str)) {
        dataSetDir = "./data/";
        std::cout << "INFO: dataSet dir is not set!\n";
    } else {
        dataSetDir = num_str;
    }
    if (!parser.getCmdOption("-refDir", num_str)) {
        refDir = "./data/";
        std::cout << "INFO: reference dir is not set!\n";
    } else {
        refDir = num_str;
    }
    std::string fileRef;
    fileRef = refDir + "pagerank_ref_tigergraph.txt";
    std::string filenameOffset = dataSetDir + files + "csr_offsets.txt";
    std::string filenameIndice = dataSetDir + files + "csr_columns.txt";

    //----------------- Text Parser --------------------------
    std::string opName;
    std::string kernelName;
    int requestLoad;
    std::string xclbinPath;
    int deviceNeeded;

    std::fstream userInput("./config.json", std::ios::in);
    if (!userInput) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    char line[1024] = {0};
    char* token;
    while (userInput.getline(line, sizeof(line))) {
        token = strtok(line, "\"\t ,}:{\n");
        while (token != NULL) {
            if (!std::strcmp(token, "operationName")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                opName = token;
            } else if (!std::strcmp(token, "kernelName")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                kernelName = token;
            } else if (!std::strcmp(token, "requestLoad")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                requestLoad = std::atoi(token);
            } else if (!std::strcmp(token, "xclbinPath")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                xclbinPath = token;
            } else if (!std::strcmp(token, "deviceNeeded")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                deviceNeeded = std::atoi(token);
            }
            token = strtok(NULL, "\"\t ,}:{\n");
        }
    }
    userInput.close();

    //----------------- Setup pageRank thread ----------------------
    xf::graph::L3::Handle::singleOP op0;
    op0.operationName = (char*)opName.c_str();
    op0.setKernelName((char*)kernelName.c_str());
    op0.requestLoad = requestLoad;
    op0.xclbinFile = (char*)xclbinPath.c_str();
    op0.deviceNeeded = deviceNeeded;

    xf::graph::L3::Handle handle0;
    handle0.addOp(op0);
    handle0.setUp();

    //----------------- Readin Graph from file ---------------------
    bool weighted = 0;
    uint32_t numVertices;
    uint32_t numEdges;
    uint32_t* offsetsCSR;
    uint32_t* indicesCSR;
    DT* weightsCSR;

    readInOffset<uint32_t>(filenameOffset, numVertices, &offsetsCSR);
    readInIndice<uint32_t, DT>(filenameIndice, weighted, numEdges, &indicesCSR, &weightsCSR);
    // readInCOO<uint32_t, DT>(filenameCOO, weighted, numVertices, numEdges,
    // &offsetsCSR, &indicesCSR, &weightsCSR);

    xf::graph::Graph<uint32_t, DT> g("CSR", numVertices, numEdges, offsetsCSR, indicesCSR, weightsCSR);

    delete[] offsetsCSR;
    delete[] indicesCSR;
    delete[] weightsCSR;
    unsigned int num_runs = 1;
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Number of nodes: " << numVertices << std::endl;
    std::cout << "INFO: Number of edges: " << numEdges << std::endl;

#ifndef BANCKMARK
    DT alpha = 0.85;
    DT tolerance = 1e-3f;
    int maxIter = 20;
#else
    DT alpha = 0.85;
    DT tolerance = 1e-3f;
    int maxIter = 500;
#endif

    (handle0.oppg)->loadGraph(g);

    //----------------- Run kernel ----------------------------------
    DT* pagerank = new DT[numVertices];

    auto f = xf::graph::L3::pageRankWeight(handle0, alpha, tolerance, maxIter, g, pagerank);
    f.wait();
    (handle0.oppg)->join();
    handle0.free();
    g.freeBuffers();

    //----------------- Readin golden value--------------------------
    DT* golden = new DT[numVertices];
    for (int i = 0; i < numVertices; ++i) {
        golden[i] = 0;
    }
    readInRef<int, DT>(fileRef, golden, numVertices);
    float sum2 = 0.0;
    for (int i = 0; i < numVertices; ++i) {
        sum2 += golden[i];
    }

    //---------------- Compare and write out results ----------------
    std::fstream fin("pagerank1.output", std::ios::out);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    float sum3 = 0.0;
    for (int i = 0; i < numVertices; ++i) {
        sum3 += pagerank[i];
        fin << i << "  ";
        fin << pagerank[i] << "\n";
    }

    fin.close();

    std::cout << "INFO: sum_golden = " << sum2 << std::endl;
    std::cout << "INFO: sum_pagerank = " << sum3 << std::endl;

    // Calculate err
    DT err = 0.0;
    int accurate = 0;
    for (int i = 0; i < numVertices; ++i) {
        err += (golden[i] - pagerank[i]) * (golden[i] - pagerank[i]);
        if (std::abs(pagerank[i] - golden[i]) < tolerance) {
            accurate += 1;
        } else {
            std::cout << "pagerank i = " << i << "\t our = " << pagerank[i] << "\t golden = " << golden[i] << std::endl;
        }
    }
    DT accRate = accurate * 1.0 / numVertices;
    err = std::sqrt(err);
    std::cout << "INFO: Accurate Rate = " << accRate << std::endl;
    std::cout << "INFO: Err Geomean = " << err << std::endl;

    delete[] pagerank;
    delete[] golden;

    if (err < numVertices * tolerance) {
        std::cout << "INFO: Result is correct" << std::endl;
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        std::cout << "INFO: Result is wrong" << std::endl;
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
