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

#define DT float

int main(int argc, const char* argv[]) {
    //--------------- cmd parser -------------------------------

    // cmd parser
    ArgParser parser(argc, argv);

    std::string filenameWeight;
    std::string filenameLabel;
    std::string goldenFile;
    std::string tmpStr;

    const int splitNm = 4; // kernel has 4 PUs, the input data should be splitted into 4 parts

    unsigned int numVertices = 16;                   // total number of vertex read from file
    unsigned int numEdges = 5;                       // total number of edge read from file
    uint32_t* numVerticesPU = new uint32_t[splitNm]; // vertex numbers in each PU
    uint32_t* numEdgesPU = new uint32_t[splitNm];    // edge numbers in each PU
    uint32_t topK;
    std::cout << "INFO: use dense graph" << std::endl;

    if (!parser.getCmdOption("-weight", tmpStr)) { // weight
        filenameWeight = "./data/knn_dense_weight.csr";
        std::cout << "INFO: indices file path is not set, use default " << filenameWeight << "\n";
    } else {
        filenameWeight = tmpStr;
        std::cout << "INFO: indices file path is " << filenameWeight << std::endl;
    }

    if (!parser.getCmdOption("-label", tmpStr)) { // weight
        filenameLabel = "./data/knn_dense_label.csr";
        std::cout << "INFO: label file path is not set, use default " << filenameLabel << "\n";
    } else {
        filenameLabel = tmpStr;
        std::cout << "INFO: label file path is " << filenameLabel << std::endl;
    }

    if (!parser.getCmdOption("-golden", tmpStr)) { // golden
        goldenFile = "./data/knn_dense.mtx";
        std::cout << "INFO: golden file path is not set!\n";
    } else {
        goldenFile = tmpStr;
        std::cout << "INFO: golden file path is " << goldenFile << std::endl;
    }

    if (!parser.getCmdOption("-topK", tmpStr)) { // topK
        topK = 32;
        std::cout << "INFO: topK is not set, use 32 by default" << std::endl;
    } else {
        topK = std::stoi(tmpStr);
    }

    //---------------- setup number of vertices in each PU ---------
    for (int i = 0; i < splitNm; ++i) {
        numVerticesPU[i] = 1;
    }

    int dataType = 1; // int32:0    float:1
                      //    int sourceID = 3; // source ID

    uint32_t* numElementsPU;
    numElementsPU = new uint32_t[splitNm];
    for (int i = 0; i < splitNm; i++) {
        numEdgesPU[i] = numEdges;
        numElementsPU[i] = numEdges * numVerticesPU[i];
    }

    xf::graph::Graph<uint32_t, DT> g("Dense", 4 * splitNm, numElementsPU);
    g.numEdgesPU = new uint32_t[splitNm];
    g.numVerticesPU = new uint32_t[splitNm];
    g.edgeNum = numEdges;
    g.nodeNum = numVertices;
    g.splitNum = splitNm;
    for (int i = 0; i < splitNm; ++i) {
        g.numVerticesPU[i] = numVerticesPU[i];
        g.numEdgesPU[i] = numEdgesPU[i];
    }

    std::fstream weightfstream(filenameWeight.c_str(), std::ios::in);
    if (!weightfstream) {
        std::cout << "Error: " << filenameWeight << "weight file doesn't exist !" << std::endl;
        exit(1);
    }

    unsigned int sumVertex = 0;
    for (int i = 0; i < splitNm; i++) { // offset32 buffers allocation
        sumVertex += 4 * numVerticesPU[i];
    }
    if (sumVertex != numVertices) { // vertex numbers between file input and numVerticesPU should match
        std::cout << "Error : sum of PU vertex numbers doesn't match file input vertex number!" << std::endl;
        exit(1);
    }

    readInWeight<splitNm>(weightfstream, dataType, numElementsPU, g.weightsDense);

    std::cout << "INFO: numVertice=" << numVertices << std::endl;
    std::cout << "INFO: numEdges=" << numEdges << std::endl;

    unsigned int sourceNm = 1;
    //---------------- Generate Source Indice and Weight Array -------
    //    unsigned int sourceLen; // sourceIndice array length
    //    uint32_t* sourceWeight; // weights of source vertex's out members
    //    generateSourceParams<splitNm>(numVerticesPU, numEdges, dataType, sourceID, g.weightsDense, sourceLen,
    //                                  &sourceWeight);

    std::fstream labelfstream(filenameLabel.c_str(), std::ios::in);
    if (!labelfstream) {
        std::cout << "Error: " << filenameLabel << "offset file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string* labels;
    readInLabel<std::string>(labelfstream, numVertices, &labels);

    //----------------- Text Parser ----------------------------------
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

    //----------------- Setup similarity thread ---------
    xf::graph::L3::Handle::singleOP op0;
    op0.operationName = (char*)opName.c_str();
    op0.setKernelName((char*)kernelName.c_str());
    op0.requestLoad = requestLoad;
    op0.xclbinFile = (char*)xclbinPath.c_str();
    op0.deviceNeeded = deviceNeeded;

    xf::graph::L3::Handle handle0;
    handle0.addOp(op0);
    handle0.setUp();

    //---------------- Load Graph -----------------------------------
    (handle0.opsimdense)->loadGraph(g);
    std::string* predictedLabel = new std::string[g.nodeNum];

    //---------------- Run L3 API -----------------------------------
    auto ev = xf::graph::L3::knnSimilarityAPDense(handle0, topK, g, labels, predictedLabel);
    int ret = ev.wait();

    (handle0.opsimdense)->join();
    handle0.free();
    g.freeBuffers();
    //---------------- Check Result ---------------------------------
    std::string golden = "white";
    int err = strcmp(predictedLabel[3].c_str(), golden.c_str());
    delete[] numElementsPU;
    delete[] numVerticesPU;
    delete[] numEdgesPU;
    delete[] labels;
    free(g.numEdgesPU);
    free(g.numVerticesPU);
    delete[] predictedLabel;

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}
