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
#include "utils.hpp"
#include "xf_graph_L3.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "stdlib.h"
#include <unordered_map>
#include "xf_utils_sw/logger.hpp"

#define DT uint32_t

double showTimeData2(std::string p_Task, TimePointType& t1, TimePointType& t2) {
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    std::cout << p_Task << "  " << std::fixed << std::setprecision(6) << l_timeMs << " msec" << std::endl;
    return l_timeMs;
};

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
    //--------------- cmd parser -------------------------------
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    std::string filenameOffset;
    std::string filenameIndice;
    std::string filenameGolden;
    std::string num_str;
    if (!parser.getCmdOption("-offset", num_str)) {
        filenameOffset = "./data/test_offsets.csr";
        std::cout << "INFO: offset file is not set!\n";
    } else {
        filenameOffset = num_str;
    }
    if (!parser.getCmdOption("-indice", num_str)) {
        filenameIndice = "./data/test_columns.csr";
        std::cout << "INFO: indice file is not set!\n";
    } else {
        filenameIndice = num_str;
    }
    if (!parser.getCmdOption("-golden", num_str)) {
        filenameGolden = "./data/test_golden.mtx";
        std::cout << "INFO: golden file is not set!\n";
    } else {
        filenameGolden = num_str;
    }
    std::cout << "INFO: offset file path is " << filenameOffset << std::endl;
    std::cout << "INFO: indice file path is " << filenameIndice << std::endl;

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

    //----------------- Setup shortestPathFloat thread ---------
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
    uint32_t num_runs = 1;
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Number of nodes: " << numVertices << std::endl;
    std::cout << "INFO: Number of edges: " << numEdges << std::endl;
    uint32_t* result = xf::graph::L3::aligned_alloc<uint32_t>(numVertices);

    //---------------- Run L3 API -----------------------------------
    auto ev = xf::graph::L3::scc(handle0, g, result);
    int ret = ev.wait();

    //---------------- Check Result ---------------------------------
    std::unordered_map<int, int> map;
    for (int i = 0; i < numVertices; i++) {
        map[result[i]] = 1;
    }
    std::cout << "INFO: HW components:" << map.size() << std::endl;

    std::vector<int> gold_result(numVertices, -1);

    std::fstream goldenfstream(filenameGolden.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error : " << filenameGolden << " file doesn't exist !" << std::endl;
        exit(1);
    }
    uint32_t index = 0;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        std::string tmp[2];
        int tmpi[2];
        data >> tmp[0];
        data >> tmp[1];

        tmpi[0] = std::stoi(tmp[0]);

        if (index > 0) {
            tmpi[1] = std::stoi(tmp[1]);
            gold_result[tmpi[0] - 1] = tmpi[1];
        } else
            std::cout << "The number of components:" << tmpi[0] << std::endl;

        index++;
    }

    if (index - 1 != numVertices) {
        std::cout << "Warning : Some nodes are missing in the golden file, validation will skip them." << std::endl;
    }

    int err = 0;
    for (int i = 0; i < numVertices; i++) {
        if (gold_result[i] != -1 && result[i] != gold_result[i]) {
            std::cout << "INFO: Mismatch-" << i << ":\tsw: " << gold_result[i] << " -> "
                      << "hw: " << result[i] << std::endl;
            err++;
        }
    }
    //--------------- Free and delete -----------------------------------
    (handle0.opscc)->join();
    handle0.free();
    g.freeBuffers();
    free(result);

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    if (err == 0) {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
