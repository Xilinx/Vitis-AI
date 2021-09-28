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
#include "xf_utils_sw/logger.hpp"

#include <cstdlib>
#include <time.h>
#include "common.hpp"

#define DT float

int main(int argc, const char* argv[]) {
    //--------------- cmd parser -------------------------------

    // cmd parser
    ArgParser parser(argc, argv);

    std::string filenameWeight;
    std::string goldenFile;
    std::string tmpStr;

    const int splitNm = 3;    // kernel has 4 PUs, the input data should be splitted into 4 parts
    const int channelsPU = 4; // each PU has 4 HBM channels
    const int cuNm = 2;       // equivalent CU numbers in one xclbin
    int deviceNeeded = 1;     // needed fpga board number
    const int channelW = 16;  // AXI interface bitwidth is 512bits, which equals to 16 x 32bits

    int numVertices = 1580; // total number of vertex read from file
    int numEdges = 200;     // total number of edge read from file
    int32_t edgeAlign8 = ((numEdges + channelW - 1) / channelW) * channelW;
    int general = ((numVertices + deviceNeeded * cuNm * splitNm * channelsPU - 1) /
                   (deviceNeeded * cuNm * splitNm * channelsPU)) *
                  channelsPU;
    int rest = numVertices - general * (deviceNeeded * cuNm * splitNm - 1);
    if (rest < 0) {
        exit(1);
    }
    int32_t** numVerticesPU = new int32_t*[deviceNeeded * cuNm]; // vertex numbers in each PU
    int32_t** numEdgesPU = new int32_t*[deviceNeeded * cuNm];    // edge numbers in each PU

    int tmpID[deviceNeeded * cuNm * channelsPU * splitNm];
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        numVerticesPU[i] = new int32_t[splitNm];
        numEdgesPU[i] = new int32_t[splitNm];
        for (int j = 0; j < splitNm; ++j) {
            numEdgesPU[i][j] = numEdges;
            for (int k = 0; k < channelsPU; ++k) {
                tmpID[i * splitNm * channelsPU + j * channelsPU + k] = 0;
            }
        }
    }
    int32_t topK;
    std::cout << "INFO: use dense graph" << std::endl;

    if (!parser.getCmdOption("-weight", tmpStr)) { // weight
        filenameWeight = "./data/cosine_dense_weight.csr";
        std::cout << "INFO: indices file path is not set, use default " << filenameWeight << "\n";
    } else {
        filenameWeight = tmpStr;
        std::cout << "INFO: indices file path is " << filenameWeight << std::endl;
    }

    if (!parser.getCmdOption("-golden", tmpStr)) { // golden
        goldenFile = "./data/cosine_sparse.mtx";
        std::cout << "INFO: golden file path is not set!\n";
    } else {
        goldenFile = tmpStr;
        std::cout << "INFO: golden file path is " << goldenFile << std::endl;
    }

    if (!parser.getCmdOption("-topK", tmpStr)) { // topK
        topK = 100;
        std::cout << "INFO: topK is not set, use 32 by default" << std::endl;
    } else {
        topK = std::stoi(tmpStr);
    }

    //----------------- Text Parser ----------------------------------
    std::string opName;
    std::string kernelName;
    int requestLoad = 100;
    std::string xclbinPath;
    std::string xclbinPath2;

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
            } else if (!std::strcmp(token, "xclbinPath2")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                xclbinPath2 = token;
            } else if (!std::strcmp(token, "deviceNeeded")) {
                token = strtok(NULL, "\"\t ,}:{\n");
                //    deviceNeeded = std::atoi(token);
            }
            token = strtok(NULL, "\"\t ,}:{\n");
        }
    }
    userInput.close();

    //----------------- Setup similarity thread ---------
    xf::graph::L3::Handle::singleOP op0;
    op0.operationName = (char*)opName.c_str();
    op0.setKernelName((char*)kernelName.c_str());
    op0.requestLoad = requestLoad;
    op0.xclbinFile = (char*)xclbinPath.c_str();
    op0.xclbinFile2 = (char*)xclbinPath2.c_str();
    op0.deviceNeeded = deviceNeeded;
    op0.cuPerBoard = cuNm;

    xf::graph::L3::Handle handle0;
    handle0.addOp(op0);
    handle0.setUp();

    //---------------- setup number of vertices in each PU ---------
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        for (int j = 0; j < splitNm; ++j) {
            numVerticesPU[i][j] = general;
        }
    }
    if (rest != 0) {
        numVerticesPU[deviceNeeded * cuNm - 1][splitNm - 1] = rest;
    }

    int sourceID = 4; // source ID

    xf::graph::Graph<int32_t, int32_t>** g = new xf::graph::Graph<int32_t, int32_t>*[deviceNeeded * cuNm];
    int fpgaNodeNm = 0;
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        g[i] = new xf::graph::Graph<int32_t, int32_t>("Dense", 4 * splitNm, numEdges, numVerticesPU[i]);
        g[i][0].numEdgesPU = new int32_t[splitNm];
        g[i][0].numVerticesPU = new int32_t[splitNm];
        g[i][0].edgeNum = numEdges;
        g[i][0].nodeNum = numVertices;
        g[i][0].splitNum = splitNm;
        g[i][0].refID = fpgaNodeNm;
        for (int j = 0; j < splitNm; ++j) {
            fpgaNodeNm += numVerticesPU[i][j];
            g[i][0].numVerticesPU[j] = numVerticesPU[i][j];
            int depth = ((numVerticesPU[i][j] + channelsPU - 1) / channelsPU) * edgeAlign8;
            g[i][0].numEdgesPU[j] = depth;
        }
    }
    std::fstream weightfstream(filenameWeight.c_str(), std::ios::in);
    if (!weightfstream) {
        std::cout << "Error: " << filenameWeight << "weight file doesn't exist !" << std::endl;
        exit(1);
    }

    int sourceLen = edgeAlign8;                                               // sourceIndice array length
    int32_t* sourceWeight = xf::graph::L3::aligned_alloc<int32_t>(sourceLen); // weights of source vertex's out members
    memset(sourceWeight, 0, sourceLen * sizeof(int32_t));

    int id = 0;
    int counter = 0;
    int row = 0;
    int splitID = 0;
    char line2[1024] = {0};
    int32_t VidChannel = (numVerticesPU[0][0] + channelsPU - 1) / channelsPU;
    int32_t cntG = 0;
    int32_t cntSrc = 0;
    int32_t srcBegin = (sourceID - 1) * numEdges;
    int32_t srcEnd = sourceID * numEdges - 1;
    int32_t currentNode = 0;
    while (weightfstream.getline(line2, sizeof(line2))) {
        std::stringstream data(line2);
        int32_t tmp;
        data >> tmp;
        g[cntG][0].weightsDense[row][tmpID[cntG * channelsPU * splitNm + row] * edgeAlign8 + id] = tmp;
        if ((currentNode >= srcBegin) && (currentNode <= srcEnd)) {
            sourceWeight[cntSrc] = tmp;
            cntSrc++;
        }
        currentNode++;
        id++;
        counter++;
        if (counter == (numVerticesPU[cntG][splitID] * numEdges)) {
            splitID++;
            counter = 0;
            row++;
            if (splitID < splitNm) {
                VidChannel = (numVerticesPU[cntG][splitID] + channelsPU - 1) / channelsPU;
            } else {
                cntG++;
                splitID = 0;
                row = 0;
                if (cntG < deviceNeeded * cuNm) {
                    VidChannel = (numVerticesPU[cntG][splitID] + channelsPU - 1) / channelsPU;
                }
            }
            id = 0;
        } else if ((tmpID[cntG * channelsPU * splitNm + row] == (VidChannel - 1)) && (id >= numEdges)) {
            id = 0;
            row++;
        } else if (id >= numEdges) {
            id = 0;
            tmpID[cntG * channelsPU * splitNm + row]++;
        }
    }
    weightfstream.close();

    //---------------- Load Graph -----------------------------------
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        (handle0.opsimdense)->loadGraphMultiCardNonBlocking(i / cuNm, i % cuNm, g[i][0]);
    }

    int32_t hwNm = deviceNeeded * cuNm;
    int32_t requestNm = 10;
    std::vector<xf::graph::L3::event<int> > eventQueue[requestNm];
    float* similarity[requestNm];
    int32_t* resultID[requestNm];
    float** similarity0[requestNm];
    int32_t** resultID0[requestNm];
    int counter0[requestNm][hwNm];
    for (int m = 0; m < requestNm; ++m) {
        similarity[m] = xf::graph::L3::aligned_alloc<float>(topK);
        resultID[m] = xf::graph::L3::aligned_alloc<int32_t>(topK);
        memset(resultID[m], 0, topK * sizeof(int32_t));
        memset(similarity[m], 0, topK * sizeof(float));

        similarity0[m] = new float*[hwNm];
        resultID0[m] = new int32_t*[hwNm];
        for (int i = 0; i < hwNm; ++i) {
            counter0[m][i] = 0;
            similarity0[m][i] = aligned_alloc<float>(topK);
            resultID0[m][i] = aligned_alloc<int32_t>(topK);
            memset(resultID0[m][i], 0, topK * sizeof(int32_t));
            memset(similarity0[m][i], 0, topK * sizeof(float));
        }
    }
    //---------------- Run L3 API -----------------------------------
    for (int m = 0; m < requestNm; ++m) {
        eventQueue[m] = xf::graph::L3::cosineSimilaritySSDenseMultiCard(handle0, hwNm, sourceLen, sourceWeight, topK, g,
                                                                        resultID0[m], similarity0[m]);
    }

    int ret = 0;
    for (int m = 0; m < requestNm; ++m) {
        for (int i = 0; i < (int)eventQueue[m].size(); ++i) {
            ret += eventQueue[m][i].wait();
        }
    }
    for (int m = 0; m < requestNm; ++m) {
        for (int i = 0; i < topK; ++i) {
            similarity[m][i] = similarity0[m][0][counter0[m][0]];
            int32_t prev = 0;
            resultID[m][i] = resultID0[m][0][counter0[m][0]];
            counter0[m][0]++;
            for (int j = 1; j < hwNm; ++j) {
                if (similarity[m][i] < similarity0[m][j][counter0[m][j]]) {
                    similarity[m][i] = similarity0[m][j][counter0[m][j]];
                    resultID[m][i] = resultID0[m][j][counter0[m][j]];
                    counter0[m][prev]--;
                    prev = j;
                    counter0[m][j]++;
                }
            }
        }
    }
    //---------------- Check Result ---------------------------------
    int err = 0;
    for (int m = 0; m < requestNm; ++m) {
        err += checkData<splitNm>(goldenFile, resultID[m], similarity[m]);
    }

    for (int m = 0; m < requestNm; ++m) {
        for (int i = 0; i < hwNm; ++i) {
            free(similarity0[m][i]);
            free(resultID0[m][i]);
        }
        delete[] similarity0[m];
        delete[] resultID0[m];
    }

    handle0.free();
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        g[i]->freeBuffers();
        delete[] g[i]->numEdgesPU;
        delete[] g[i]->numVerticesPU;
    }
    delete[] g;
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        delete[] numVerticesPU[i];
        delete[] numEdgesPU[i];
    }
    free(sourceWeight);
    delete[] numVerticesPU;
    delete[] numEdgesPU;
    for (int m = 0; m < requestNm; ++m) {
        free(similarity[m]);
        free(resultID[m]);
    }
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}
