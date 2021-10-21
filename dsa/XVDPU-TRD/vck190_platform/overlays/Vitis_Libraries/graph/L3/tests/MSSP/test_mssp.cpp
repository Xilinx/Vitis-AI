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

#define DT float

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
    std::string files;
    std::string dataSetDir;
    std::string refDir;
    std::string num_str;
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
    fileRef = refDir + "sssp_golden.txt";
    std::string filenameOffset = dataSetDir + files + "csr_offsets.txt";
    std::string filenameIndice = dataSetDir + files + "csr_columns.txt";
    std::string filenameCOO = dataSetDir + files + "coo.txt";
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
    bool weighted = 1;
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

    uint32_t nSource = 10;
    uint32_t* sourceID;
    sourceID = new uint32_t[nSource];

    //---------------- Find max OutDegree source ID ----------------
    int max = 0;
    int id = 0;
    for (int i = 0; i < numVertices; i++) {
        if (g.offsetsCSR[i + 1] - g.offsetsCSR[i] > max) {
            max = g.offsetsCSR[i + 1] - g.offsetsCSR[i];
            id = i;
        }
    }
    std::cout << "INFO: source ID " << id << "\t the max outDegree is " << max << std::endl;
    for (int i = 0; i < nSource; ++i) {
        sourceID[i] = id;
    }

    //---------------- Load Graph -----------------------------------
    (handle0.opsp)->loadGraph(g);

    uint32_t length = ((numVertices + 1023) / 1024) * 1024;
    DT** result;
    uint32_t** pred;
    result = new DT*[nSource];
    pred = new uint32_t*[nSource];
    for (int i = 0; i < nSource; ++i) {
        result[i] = xf::graph::L3::aligned_alloc<DT>(length);
        pred[i] = xf::graph::L3::aligned_alloc<uint32_t>(length);
        memset(result[i], 0, length * sizeof(DT));
        memset(pred[i], 0, length * sizeof(uint32_t));
    }

    //---------------- Run L3 API -----------------------------------
    auto ev = xf::graph::L3::shortestPath(handle0, nSource, sourceID, weighted, g, result, pred);
    int ret = ev.wait();

    //---------------- Check Result ---------------------------------
    int err = 0;
    for (int k = 0; k < nSource; ++k) {
        bool* connect;
        connect = xf::graph::L3::aligned_alloc<bool>(length);

        for (int i = 0; i < numVertices; i++) {
            connect[i] = false;
        }

        std::fstream goldenfstream(fileRef.c_str(), std::ios::in);
        if (!goldenfstream) {
            std::cout << "Error : " << fileRef << " file doesn't exist !" << std::endl;
            exit(1);
        }
        char line[1024] = {0};
        goldenfstream.getline(line, sizeof(line));

        int index = 0;
        while (goldenfstream.getline(line, sizeof(line))) {
            std::string str(line);
            std::replace(str.begin(), str.end(), ',', ' ');
            std::stringstream data(str.c_str());
            int vertex;
            float distance;
            int pred_golden;
            data >> vertex;
            data >> distance;
            data >> pred_golden;
            if (std::abs(result[k][vertex - 1] - distance) / distance > 0.00001) {
                std::cout << "Error : distance " << vertex - 1 << " " << distance << " " << result[k][vertex - 1]
                          << std::endl;
                err++;
            }
            if (pred_golden - 1 != pred[k][vertex - 1]) {
                unsigned int tmp_fromID = pred[k][vertex - 1];
                unsigned int tmp_toID = vertex - 1;
                float tmp_distance = 0;
                int iter = 0;
                while ((tmp_fromID != sourceID[0] || tmp_toID != sourceID[0]) && iter < numVertices) {
                    float tmp_weight = 0;
                    int begin = g.offsetsCSR[tmp_fromID];
                    int end = g.offsetsCSR[tmp_fromID + 1];
                    for (int i = begin; i < end; i++) {
                        if (g.indicesCSR[i] == tmp_toID) {
                            tmp_weight = g.weightsCSR[i];
                        }
                    }
                    tmp_distance = tmp_distance + tmp_weight;
                    tmp_toID = tmp_fromID;
                    tmp_fromID = pred[k][tmp_fromID];
                    iter++;
                }
                if (std::abs(result[k][vertex - 1] - tmp_distance) / tmp_distance > 0.00001) {
                    std::cout << "Error : predecessor " << vertex - 1 << std::endl;
                    tmp_fromID = pred[k][vertex - 1];
                    tmp_toID = vertex - 1;
                    iter = 0;
                    while ((tmp_fromID != sourceID[0] || tmp_toID != sourceID[0]) && iter < numVertices) {
                        tmp_toID = tmp_fromID;
                        tmp_fromID = pred[k][tmp_fromID];
                        iter++;
                    }
                    err++;
                }
            }
            connect[vertex - 1] = true;
        }

        for (int i = 0; i < numVertices; i++) {
            if (connect[i] == false && result[k][i] != std::numeric_limits<float>::infinity()) {
                std::cout << "Error : distance " << i << " " << std::numeric_limits<float>::infinity() << " "
                          << result[k][i] << std::endl;
                err++;
            }
            if (connect[i] == false && pred[k][i] != std::numeric_limits<unsigned int>::max()) {
                std::cout << "Error : predecessor " << i << " not connected " << pred[k][i] << std::endl;
                err++;
            }
        }
        if (err == 0) {
            std::cout << "INFO: Test passed." << std::endl;
        } else {
            std::cout << "Error: There are in total " << err << " errors." << std::endl;
        }
        free(connect);
    }

    //--------------- Free and delete -----------------------------------
    (handle0.opsp)->join();
    handle0.free();
    g.freeBuffers();

    delete[] sourceID;
    for (int i = 0; i < nSource; ++i) {
        free(result[i]);
        free(pred[i]);
    }
    delete[] result;
    delete[] pred;

    return err;
}
