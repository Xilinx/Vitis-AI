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
    ArgParser parser(argc, argv);

    std::string filenameOffset;
    std::string filenameIndice;
    std::string goldenFile;
    std::string tmpStr;

    const int splitNm = 8; // kernel has 4 PUs, the input data should be splitted into 4 parts

    unsigned int numVertices = 16;                   // total number of vertex read from file
    unsigned int numEdges = 5;                       // total number of edge read from file
    uint32_t* numVerticesPU = new uint32_t[splitNm]; // vertex numbers in each PU
    uint32_t* numEdgesPU = new uint32_t[splitNm];    // edge numbers in each PU
    uint32_t topK;
    std::cout << "INFO: use dense graph" << std::endl;

    if (!parser.getCmdOption("-offset", tmpStr)) { // weight
        filenameOffset = "./data/cosine_sparse_offset.csr";
        std::cout << "INFO: indices file path is not set, use default " << filenameOffset << "\n";
    } else {
        filenameOffset = tmpStr;
        std::cout << "INFO: indices file path is " << filenameOffset << std::endl;
    }

    if (!parser.getCmdOption("-indiceWeight", tmpStr)) { // weight
        filenameIndice = "./data/cosine_sparse_indice_weight.csr";
        std::cout << "INFO: indices file path is not set, use default " << filenameIndice << "\n";
    } else {
        filenameIndice = tmpStr;
        std::cout << "INFO: indices file path is " << filenameIndice << std::endl;
    }

    if (!parser.getCmdOption("-golden", tmpStr)) { // golden
        goldenFile = "./data/cosine_sparse.mtx";
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

    //---------------- setup number of vertices in each PU ---------------
    for (int i = 0; i < splitNm - 1; ++i) {
        numVerticesPU[i] = 4;
    }
    numVerticesPU[splitNm - 1] = 6; // vertex number in the last PU

    int dataType = 1; // int32:0    float:1

    //----------- read in numVertices numEdges from files ----------------
    std::fstream offsetfstream(filenameOffset.c_str(), std::ios::in);
    if (!offsetfstream) {
        std::cout << "Error: " << filenameOffset << "offset file doesn't exist !" << std::endl;
        exit(1);
    }
    readInConstNUM(offsetfstream, numVertices);

    std::fstream indicefstream(filenameIndice.c_str(), std::ios::in);
    if (!indicefstream) {
        std::cout << "Error: " << filenameIndice << "indice && weight file doesn't exist !" << std::endl;
        exit(1);
    }
    readInConstNUM(indicefstream, numEdges);

    xf::graph::Graph<uint32_t, DT> g("CSR", numVertices, numEdges);
    g.allocSplittedOffsets(splitNm, numVerticesPU);

    g.numVerticesPU[splitNm - 1] += 1;
    //--------------------- read in data from file ------------------------
    readInOffset<uint32_t, splitNm>(offsetfstream, g.numVerticesPU, g.offsetsSplitted);

    for (int i = 0; i < splitNm - 1; ++i) { // add datas in order to meet kernel need
        g.offsetsSplitted[i][numVerticesPU[i]] = g.offsetsSplitted[i + 1][0];
    }
    g.numVerticesPU[splitNm - 1] -= 1; // adjust last numVerticesPU in order to meet kernel need

    g.allocSplittedIndicesWeights();

    readInIndiceWeight<splitNm>(indicefstream, dataType, g.numEdgesPU, g.indicesSplitted, g.weightsSplitted);

    std::cout << "INFO: numVertices=" << numVertices << std::endl;
    std::cout << "INFO: numEdges=" << numEdges << std::endl;

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
    (handle0.opsimsparse)->loadGraph(g);
    float** similarity;
    uint32_t** resultID;
    similarity = new float*[numVertices];
    resultID = new uint32_t*[numVertices];
    for (int i = 0; i < numVertices; ++i) {
        similarity[i] = xf::graph::L3::aligned_alloc<float>(topK);
        resultID[i] = xf::graph::L3::aligned_alloc<uint32_t>(topK);
        memset(resultID[i], 0, topK * sizeof(uint32_t));
        memset(similarity[i], 0, topK * sizeof(float));
    }

    //---------------- Run L3 API -----------------------------------
    auto ev = xf::graph::L3::cosineSimilarityAPSparse(handle0, topK, g, resultID, similarity);
    int ret = ev.wait();

    //---------------- Check Result ---------------------------------
    int err = checkData<splitNm>(goldenFile, resultID[3], similarity[3]);

    (handle0.opsimsparse)->join();
    handle0.free();
    g.freeBuffers();
    delete[] numVerticesPU;
    delete[] numEdgesPU;
    delete[] g.numEdgesPU;
    delete[] g.numVerticesPU;

    for (int i = 0; i < numVertices; ++i) {
        free(similarity[i]);
        free(resultID[i]);
    }
    delete[] similarity;
    delete[] resultID;

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}
