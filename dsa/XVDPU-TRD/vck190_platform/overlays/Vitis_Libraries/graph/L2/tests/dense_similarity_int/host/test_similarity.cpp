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
    std::string offsetFile;
    std::string indiceFile;
    std::string weightFile;
    std::string goldenFile;
    std::string tmpStr;

    int repInt = 1;         // kernel repeat numbers
    int similarityType = 0; // jaccard:0  cosine:1
    int dataType = 0;       // int32:0    float:1
    int graphType = 0;      // sparse:0   dense:1
    int sourceID = 0;       // source ID
    int sortK = 32;         // topK

    unsigned int numVertices = 160;        // total number of vertex read from file
    unsigned int numEdges = 200;           // total number of edge read from file
    unsigned int numVerticesPU[PU_NUMBER]; // vertex numbers in each PU
    unsigned int numEdgesPU[PU_NUMBER];    // edge numbers in each PU
    ap_int<32>* offset32[PU_NUMBER];       // offset arrays of multi-PUs
    ap_int<32>* indice32[PU_NUMBER];       // indice arrays of multi-PUs
    float* weightSparse[PU_NUMBER];        // weight arrays of sparse PUs
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

    if (graphType == 0) {
        std::cout << "INFO: use sparse graph" << std::endl;
        if (!parser.getCmdOption("-offset", offsetFile)) { // offset
            std::cout << "INFO: offset file path is not set, use default " << offsetFile << "\n";
        } else {
            std::cout << "INFO: offset file path is " << offsetFile << std::endl;
        }

        if (!parser.getCmdOption("-indiceWeight", indiceFile)) { // indice && weight
            std::cout << "INFO: indices file path is not set, use default " << indiceFile << "\n";
        } else {
            std::cout << "INFO: indices file path is " << indiceFile << std::endl;
        }

    } else {
        std::cout << "INFO: use dense graph" << std::endl;
        if (!parser.getCmdOption("-weight", weightFile)) { // weight
            std::cout << "INFO: weight file path is not set, use default " << weightFile << "\n";
        } else {
            std::cout << "INFO: weight file path is " << weightFile << std::endl;
        }

        if (!parser.getCmdOption("-vertices", tmpStr)) { // numVertices
            std::cout << "INFO: vertice number is not set, use default " << numVertices << "\n";
        } else {
            numVertices = std::stoi(tmpStr);
        }

        if (!parser.getCmdOption("-edges", tmpStr)) { // numEdges
            std::cout << "INFO: vertice number is not set, use default " << numVertices << "\n";
        } else {
            numEdges = std::stoi(tmpStr);
        }
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

    if (!parser.getCmdOption("-sourceID", tmpStr)) { // sourceID
        std::cout << "INFO: sourceID is not set, use 0 by default" << std::endl;
    } else {
        sourceID = std::stoi(tmpStr);
    }

    if (!parser.getCmdOption("-topK", tmpStr)) { // topK
        std::cout << "INFO: topK is not set, use 32 by default" << std::endl;
    } else {
        sortK = std::stoi(tmpStr);
    }

    // set the numbers of vertex in each PU
    if (graphType == 0) {
        for (int i = 0; i < PU_NUMBER - 1; ++i) {
            numVerticesPU[i] = 4;
        }
        numVerticesPU[PU_NUMBER - 1] = 6; // vertex number in the last PU
    } else {
        for (int i = 0; i < PU_NUMBER; ++i) {
            numVerticesPU[i] = 10;
        }
    }

    // read in data from file
    readInDataFile<PU_NUMBER>(offsetFile, indiceFile, weightFile, graphType, dataType, numVerticesPU, numEdgesPU,
                              numVertices, numEdges, offset32, indice32, weightSparse, weightDense);

    // generate source vertex's indice array and weight array
    unsigned int sourceNUM;   // sourceIndice array length
    ap_int<32>* sourceIndice; // source vertex's out members
    ap_int<32>* sourceWeight; // weights of source vertex's out members
    if (graphType == 0)
        generateSourceParams<PU_NUMBER>(numVertices, numEdges, dataType, sourceID, offset32, indice32, weightSparse,
                                        sourceNUM, &sourceIndice, &sourceWeight);
    else
        generateSourceParams<PU_NUMBER>(numVerticesPU, numEdges, dataType, sourceID, weightDense, sourceNUM,
                                        &sourceWeight);

    // calculate similarity
    int info;
    if (graphType == 0)
        info = computeSimilarity<PU_NUMBER>(xclbinPath, goldenFile, numVertices, numEdges, similarityType, dataType,
                                            sourceID, sortK, repInt, numVerticesPU, numEdgesPU, offset32, indice32,
                                            weightSparse, sourceNUM, sourceIndice, sourceWeight);
    else
        info = computeSimilarity<PU_NUMBER>(xclbinPath, goldenFile, numVertices, numEdges, similarityType, dataType,
                                            sourceID, sortK, repInt, numVerticesPU, numEdgesPU, weightDense, sourceNUM,
                                            sourceWeight);

    return info;
}
