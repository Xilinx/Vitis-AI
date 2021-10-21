/*
 * Copyright 2019 Xilinx, Inc.
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

#include "test_similarity.hpp"

int main(int argc, const char* argv[]) {
    std::cout << "\n-----------------Similarity----------------\n";

    // cmd parser
    ArgParser parser(argc, argv);

    std::string xclbinPath;
    std::string sourceFile;
    std::string weightFile;
    std::string goldenFile;
    std::string tmpStr;

    int repInt = 1;         // kernel repeat numbers
    int similarityType = 0; // jaccard:0  cosine:1
    int dataType = 0;       // int32:0    float:1
    int graphType = 0;      // sparse:0   dense:1
    int batchNUM = 6;       // source batch num
    int sortK = 32;         // topK

    unsigned int numVertices = 160;        // total number of vertex read from file
    unsigned int numEdges = 200;           // total number of edge read from file
    unsigned int numVerticesPU[PU_NUMBER]; // vertex numbers in each PU
    unsigned int numEdgesPU[PU_NUMBER];    // edge numbers in each PU
    float* weightDense[4 * PU_NUMBER];     // weight arrays of dense PUs

#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbinPath)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
#endif

    if (!parser.getCmdOption("-graphType", tmpStr)) { // graphType
        std::cout << "INFO: graph type is not set, use sparse graph" << std::endl;
    } else {
        graphType = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-similarityType", tmpStr)) { // similarityType
        std::cout << "INFO: similarity type is not set, use jaccard similarity by default" << std::endl;
    } else {
        similarityType = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-dataType", tmpStr)) { // dataType
        std::cout << "INFO: data type is not set, use int32 by default" << std::endl;
    } else {
        dataType = std::stoi(tmpStr);
    }

    std::cout << "INFO: use dense graph" << std::endl;
    if (!parser.getCmdOption("-weight", weightFile)) { // weight
        std::cout << "INFO: weight file path is not set, use default " << weightFile << "\n";
    } else {
        std::cout << "INFO: weight file path is " << weightFile << std::endl;
    }

    std::cout << "INFO: use batch mode" << std::endl;
    if (!parser.getCmdOption("-source", sourceFile)) { // weight
        std::cout << "INFO: source file path is not set, use default " << sourceFile << "\n";
    } else {
        std::cout << "INFO: source file path is " << sourceFile << std::endl;
    }

    if (!parser.getCmdOption("-vertices", tmpStr)) { // numVertices
        std::cout << "INFO: vertice number is not set, use default " << numVertices << "\n";
    } else {
        numVertices = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-edges", tmpStr)) { // numEdges
        std::cout << "INFO: edge number is not set, use default " << numEdges << "\n";
    } else {
        numEdges = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-golden", goldenFile)) { // golden
        std::cout << "INFO: golden file path is not set!\n";
    } else {
        std::cout << "INFO: golden file path is " << goldenFile << std::endl;
    }

    if (!parser.getCmdOption("-runs", tmpStr)) { // repeat num
        std::cout << "INFO: number of runs is not given, use 1 by default" << std::endl;
    } else {
        repInt = std::stoi(tmpStr);
        std::cout << "INFO: number of runs is " << repInt << std::endl;
    }

    if (!parser.getCmdOption("-batchNUM", tmpStr)) { // source batch num
        std::cout << "INFO: BatchNUM is not set, use 0 by default" << std::endl;
    } else {
        batchNUM = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-topK", tmpStr)) { // topK
        std::cout << "INFO: topK is not set, use 32 by default" << std::endl;
    } else {
        sortK = std::stoi(tmpStr);
    }

    // set the numbers of vertex in each PU
    for (int i = 0; i < PU_NUMBER; ++i) {
        numVerticesPU[i] = numVertices / 4;
    }

    // read in data from file
    float* sourceWeight; // weights of source vertex's out members
    readInDataFile<PU_NUMBER>(sourceFile, weightFile, graphType, dataType, batchNUM, numVertices, numEdges,
                              numVerticesPU, numEdgesPU, &sourceWeight, weightDense);

    // calculate similarity
    int info;
    info = computeSimilarity<PU_NUMBER>(xclbinPath, goldenFile, numVertices, numEdges, similarityType, dataType,
                                        batchNUM, sortK, repInt, numVerticesPU, numEdgesPU, weightDense, sourceWeight);

    return info;
}
