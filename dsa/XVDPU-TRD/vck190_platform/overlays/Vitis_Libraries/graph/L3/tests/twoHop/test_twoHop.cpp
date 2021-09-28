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
#include "xf_utils_sw/logger.hpp"
#include "utils.hpp"
#include "xf_graph_L3.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <unordered_map>

#define DT float

inline int tvdiff(struct timeval* tv0, struct timeval* tv1) {
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}

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
    std::cout << "\n---------------------Two Hop----------------\n";
    //--------------- cmd parser -------------------------------
    ArgParser parser(argc, argv);
    std::string offsetfile;
    std::string indexfile;
    std::string pairfile;
    std::string goldenfile;

    if (!parser.getCmdOption("--offset", offsetfile)) {
        std::cout << "ERROR: offset file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--index", indexfile)) {
        std::cout << "ERROR: index file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--pair", pairfile)) {
        std::cout << "ERROR: pair file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--golden", goldenfile)) {
        std::cout << "ERROR: golden file path is not set!\n";
        return -1;
    }

    //----------------- Text Parser --------------------------
    std::string opName;
    std::string kernelName;
    int requestLoad;
    std::string xclbinPath;
    int deviceNeeded;

    std::fstream userInput("./config.json", std::ios::in);
    if (!userInput) {
        std::cout << "Error : config file doesn't exist !" << std::endl;
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
    op0.cuPerBoard = 5;

    xf::graph::L3::Handle handle0;
    handle0.addOp(op0);
    handle0.setUp();

    //----------------- Readin Graph from file ---------------------
    //    int fileIdx = 0;

    bool weighted = 0;
    uint32_t numVertices;
    uint32_t numEdges;
    uint32_t numPairs;
    uint32_t* offsetsCSR;
    uint32_t* indicesCSR;
    DT* weightsCSR;

    readInOffset<uint32_t>(offsetfile, numVertices, &offsetsCSR);
    readInIndice<uint32_t, DT>(indexfile, weighted, numEdges, &indicesCSR, &weightsCSR);

    int fileIdx = 0;
    std::fstream pairfstream(pairfile.c_str(), std::ios::in);
    if (!pairfstream) {
        std::cout << "Error : " << pairfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    pairfstream.getline(line, sizeof(line));
    std::stringstream numPdata(line);
    numPdata >> numPairs;

    uint64_t* pair = xf::graph::L3::aligned_alloc<uint64_t>(numPairs);

    while (pairfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        uint64_t src;
        uint64_t des;
        data >> src;
        src = src - 1;
        data >> des;
        des = des - 1;
        uint64_t tmp = 0UL | (src << 32UL) | (des);
        pair[fileIdx] = tmp;
        fileIdx++;
    }
    pairfstream.close();

    xf::graph::Graph<uint32_t, DT> g("CSR", numVertices, numEdges, offsetsCSR, indicesCSR, weightsCSR);

    delete[] offsetsCSR;
    delete[] indicesCSR;
    delete[] weightsCSR;
    uint32_t num_runs = 1;
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Number of nodes: " << numVertices << std::endl;
    std::cout << "INFO: Number of edges: " << numEdges << std::endl;

    //-------------------------- load graph ------------------------------------
    struct timeval start_time_api, end_time_api, start_time_ld, end_time_ld;
    const int cuNm = 5; // equivalent CU numbers in one xclbin

    gettimeofday(&start_time_ld, 0);
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        (handle0.optwohop)->loadGraph(i / cuNm, i % cuNm, g);
    }
    gettimeofday(&end_time_ld, 0);
    std::cout << "load graph time " << tvdiff(&start_time_ld, &end_time_ld) / 1000.0 / 1000.0 << " s" << std::endl;

    uint32_t* numPart = (uint32_t*)malloc(5 * sizeof(uint32_t));

    numPart[0] = numPairs / 5;
    numPart[1] = numPairs / 5;
    numPart[2] = numPairs / 5;
    numPart[3] = numPairs / 5;
    numPart[4] = numPairs / 5 + numPairs % 5;

    uint64_t** pairPart = new uint64_t*[5];
    uint32_t** cntResPart = new uint32_t*[5];
    for (int i = 0; i < 5; i++) {
        pairPart[i] = xf::graph::L3::aligned_alloc<uint64_t>(numPart[i]);
        memcpy(pairPart[i], &pair[i * (numPairs / 5)], numPart[i] * sizeof(uint64_t));
        cntResPart[i] = xf::graph::L3::aligned_alloc<uint32_t>(numPart[i]);
    }

    //---------------- Run L3 API -----------------------------------
    gettimeofday(&start_time_api, 0);
    auto ev = xf::graph::L3::twoHop(handle0, numPart, pairPart, cntResPart, g);
    int ret = ev.wait();
    gettimeofday(&end_time_api, 0);
    std::cout << "API Execution time " << tvdiff(&start_time_api, &end_time_api) / 1000.0 / 1000.0 << " s" << std::endl;

    //--------------- check result-----------------------------------
    uint32_t* cnt_res = xf::graph::L3::aligned_alloc<uint32_t>(numPairs);
    for (int i = 0; i < 5; i++) {
        memcpy(&cnt_res[i * (numPairs / 5)], cntResPart[i], numPart[i] * sizeof(uint32_t));
    }

    int err = 0;

    std::fstream goldenfstream(goldenfile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error : " << goldenfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    std::unordered_map<unsigned long, float> goldenHashMap;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::string str(line);
        std::replace(str.begin(), str.end(), ',', ' ');
        std::stringstream data(str.c_str());
        unsigned long golden_src;
        unsigned long golden_des;
        unsigned golden_res;
        data >> golden_src;
        data >> golden_des;
        data >> golden_res;
        unsigned long tmp = 0UL | golden_src << 32UL | golden_des;
        goldenHashMap.insert(std::pair<unsigned long, unsigned>(tmp, golden_res));
    }
    goldenfstream.close();

    std::unordered_map<unsigned long, float> resHashMap;
    for (int i = 0; i < numPairs; i++) {
        unsigned long tmp_src = pair[i] / (1UL << 32UL) + 1UL;
        unsigned long tmp_des = pair[i] % (1UL << 32UL) + 1UL;
        unsigned long tmp_res = cnt_res[i];
        unsigned long tmp = 0UL | (tmp_src << 32UL) | tmp_des;
        resHashMap.insert(std::pair<unsigned long, unsigned>(tmp, tmp_res));
    }

    if (resHashMap.size() != goldenHashMap.size()) std::cout << "miss pairs!" << std::endl;
    for (auto it = resHashMap.begin(); it != resHashMap.end(); it++) {
        unsigned long tmp_src = (it->first) / (1UL << 32UL);
        unsigned long tmp_des = (it->first) % (1UL << 32UL);
        unsigned long tmp_res = it->second;
        auto got = goldenHashMap.find(it->first);
        if (got == goldenHashMap.end()) {
            std::cout << "ERROR: pair not found! cnt_src: " << tmp_src << " cnt_des: " << tmp_des
                      << " cnt_res: " << tmp_res << std::endl;
            err++;
        } else if (got->second != it->second) {
            std::cout << "ERROR: incorrect count! golden_src: " << (got->first) / (1UL << 32UL)
                      << " golden_des: " << (got->first) % (1UL << 32UL) << " golden_res: " << (got->second)
                      << " cnt_src: " << tmp_src << " cnt_des: " << tmp_des << " cnt_res: " << tmp_res << std::endl;
            err++;
        }
    }

    //--------------- Free and delete -----------------------------------
    (handle0.optwohop)->join();
    handle0.free();
    g.freeBuffers();

    for (int i = 0; i < 5; ++i) {
        free(pairPart[i]);
        free(cntResPart[i]);
    }
    delete[] pairPart;
    delete[] cntResPart;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}
