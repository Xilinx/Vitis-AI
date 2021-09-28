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

#ifndef XF_GRAPH_UTILS_HPP
#define XF_GRAPH_UTILS_HPP

#include "ap_int.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <unordered_map>

#include "generalSimilarityKernel.hpp"

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

#define XCL_BANK0 XCL_BANK(0)
#define XCL_BANK1 XCL_BANK(1)
#define XCL_BANK2 XCL_BANK(2)

#define XCL_BANK4 XCL_BANK(4)
#define XCL_BANK5 XCL_BANK(5)
#define XCL_BANK6 XCL_BANK(6)
#define XCL_BANK7 XCL_BANK(7)
#define XCL_BANK8 XCL_BANK(8)
#define XCL_BANK9 XCL_BANK(9)
#define XCL_BANK10 XCL_BANK(10)
#define XCL_BANK11 XCL_BANK(11)
#define XCL_BANK12 XCL_BANK(12)
#define XCL_BANK13 XCL_BANK(13)
#define XCL_BANK14 XCL_BANK(14)
#define XCL_BANK15 XCL_BANK(15)
#define XCL_BANK16 XCL_BANK(16)
#define XCL_BANK17 XCL_BANK(17)
#define XCL_BANK18 XCL_BANK(18)
#define XCL_BANK19 XCL_BANK(19)
#define XCL_BANK20 XCL_BANK(20)
#define XCL_BANK21 XCL_BANK(21)
#define XCL_BANK22 XCL_BANK(22)
#define XCL_BANK23 XCL_BANK(23)
#define XCL_BANK24 XCL_BANK(24)
#define XCL_BANK25 XCL_BANK(25)
#define XCL_BANK26 XCL_BANK(26)
#define XCL_BANK27 XCL_BANK(27)
#define XCL_BANK28 XCL_BANK(28)
#define XCL_BANK29 XCL_BANK(29)
#define XCL_BANK30 XCL_BANK(30)
#define XCL_BANK31 XCL_BANK(31)

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

// Compute time difference
unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

void readInConstNUM(std::fstream& fstream, unsigned int& numConst) {
    char line[1024] = {0};
    fstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numConst;
}

void readInConstNUM(std::fstream& fstream, unsigned int& numV, unsigned int& numE) {
    char line[1024] = {0};
    fstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numV;
    numOdata >> numE;
}

template <typename DT, int PUNUM>
void readInOffset(std::fstream& fstream,      // input: file stream
                  unsigned int length[PUNUM], // input: value numbers of each buffer
                  DT* buffer[PUNUM]) {        // output: output buffers
    int id = 0;
    int row = 0;
    char line[1024] = {0};
    if (!fstream) {
        for (int i = 0; i < PUNUM; ++i) {
            for (int j = 0; j < length[i] + 1; ++j) {
                buffer[i][j] = 0;
            }
        }
    } else {
        while (fstream.getline(line, sizeof(line))) {
            std::stringstream data(line);
            data >> buffer[row][id];
            // std::cout << "offset[" << row << "][" << id << "]=" << buffer[row][id] << std::endl;
            id++;
            if (id > length[row] && row < PUNUM - 1) {
                buffer[row + 1][0] = buffer[row][id - 1];
                // std::cout << "offset[" << row+1 << "][0]=" << buffer[row + 1][0] << std::endl;
                id = 1;
                row++;
            }
        }
    }
}

template <int PUNUM>
void readInWeight(std::fstream& fstream,      // input: file stream
                  int dataType,               // 0:int32, 1:float
                  unsigned int length[PUNUM], // input: value numbers of each buffer
                  float* buffer[4 * PUNUM]) { // output: output buffers
    int id = 0;
    int row = 0;
    char line[1024] = {0};
    if (!fstream) {
        for (int i = 0; i < 4 * PUNUM; ++i) {
            for (int j = 0; j < length[i / 4]; ++j) {
                buffer[i][j] = 1;
            }
        }
    } else {
        while (fstream.getline(line, sizeof(line))) {
            std::stringstream data(line);
            float tmp;
            data >> tmp;
            if (dataType == 0) {
                buffer[row][id] = bitsToFloat<int32_t, float>((int32_t)tmp);
            } else {
                buffer[row][id] = tmp;
            }
            id++;
            if (id >= length[row / 4]) {
                id = 0;
                row++;
            }
        }
    }
}

template <int PUNUM>
void readInIndiceWeight(std::fstream& fstream, // input: file stream
                        int dataType,
                        unsigned int length[PUNUM],  // input: value numbers of each buffer
                        ap_uint<32>* buffer1[PUNUM], // output: first array
                        float* buffer2[PUNUM]) {     // output: second array
    int id = 0;
    int row = 0;
    std::string line;
    bool flag = 0;
    bool flag1 = 0;
    while (std::getline(fstream, line)) {
        if (flag == 0) {
            flag = 1;
            std::string::size_type idx = line.find("\t");
            if (idx == std::string::npos) {
                flag1 = 1;
            }
        }
        std::stringstream data(line);
        data >> buffer1[row][id];
        if (flag1) {
            if (dataType == 0)
                buffer2[row][id] = bitsToFloat<int32_t, float>(1);
            else
                buffer2[row][id] = 1.0;
        } else {
            float tmp;
            data >> tmp;
            if (dataType == 0)
                buffer2[row][id] = bitsToFloat<int32_t, float>((int32_t)tmp);
            else
                buffer2[row][id] = tmp;
        }
        id++;
        if (id >= length[row]) {
            id = 0;
            row++;
        }
    }
}

template <int PUNUM>
void readInDataFile(std::string offsetFile,
                    std::string indiceFile,
                    std::string weightFile,
                    int graphType,
                    int dataType,
                    unsigned int numVerticesPU[PUNUM],
                    unsigned int numEdgesPU[PUNUM],
                    unsigned int& numVertices,
                    unsigned int& numEdges,
                    ap_uint<32>* offset32[PUNUM],
                    ap_uint<32>* indice32[PUNUM],
                    float* weightSparse[PUNUM],
                    float* weightDense[4 * PUNUM]) {
    if (graphType == 0) {
        // read in numVertices numEdges from files ////////////////////////
        std::fstream offsetfstream(offsetFile.c_str(), std::ios::in);
        if (!offsetfstream) {
            std::cout << "Error: " << offsetFile << "offset file doesn't exist !" << std::endl;
            exit(1);
        }
        readInConstNUM(offsetfstream, numVertices);

        std::fstream indicefstream(indiceFile.c_str(), std::ios::in);
        if (!indicefstream) {
            std::cout << "Error: " << indiceFile << "indice && weight file doesn't exist !" << std::endl;
            exit(1);
        }
        readInConstNUM(indicefstream, numEdges);
        // read in offset data///////////////////////////////////////////
        unsigned int sumVertex = 0;
        for (int i = 0; i < PUNUM; i++) { // offset32 buffers allocation
            offset32[i] = aligned_alloc<ap_uint<32> >(EXT_MEM_SZ);
            sumVertex += numVerticesPU[i];
        }
        if (sumVertex != numVertices) { // vertex numbers between file input and numVerticesPU should match
            std::cout << "Error : sum of PU vertex numbers doesn't match file input vertex number!" << std::endl;
            std::cout << "sumVertex =" << sumVertex << " numVertices=" << numVertices << std::endl;
            exit(1);
        }
        // read in data from file
        readInOffset<ap_uint<32>, PUNUM>(offsetfstream, numVerticesPU, offset32);
        std::cout << "INFO: numVertice=" << numVertices << std::endl;

        // calculate numEdgesPU from offset arrays/////////////////////////////
        for (int i = 0; i < PUNUM; i++) {
            if (i < PUNUM - 1) {
                numEdgesPU[i] = offset32[i + 1][0] - offset32[i][0];
            } else {
                numEdgesPU[PUNUM - 1] = numEdges - offset32[PUNUM - 1][0];
            }
        }

        // read in indices & weights///////////////////////////////////////////
        for (int i = 0; i < PUNUM; i++) {
            indice32[i] = aligned_alloc<ap_uint<32> >(EXT_MEM_SZ);
            weightSparse[i] = aligned_alloc<float>(EXT_MEM_SZ);
        }
        readInIndiceWeight<PUNUM>(indicefstream, dataType, numEdgesPU, indice32, weightSparse);
        std::cout << "INFO: numEdges=" << numEdges << std::endl;

        for (int i = 0; i < PUNUM; i++) {
            std::cout << "numVerticesPU[" << i << "]=" << numVerticesPU[i] << " numEdgesPU[" << i
                      << "]=" << numEdgesPU[i] << std::endl;
        }
    } else {
        // check files exist//////////////////////////////////////////////////
        std::fstream indicefstream(weightFile.c_str(), std::ios::in);
        if (!indicefstream) {
            std::cout << "Error: " << weightFile << "weight file doesn't exist !" << std::endl;
            exit(1);
        }

        unsigned int sumVertex = 0;
        for (int i = 0; i < PUNUM; i++) { // offset32 buffers allocation
            sumVertex += 4 * numVerticesPU[i];
        }
        if (sumVertex != numVertices) { // vertex numbers between file input and numVerticesPU should match
            std::cout << "Error : sum of PU vertex numbers doesn't match file input vertex number!" << std::endl;
            std::cout << "sumVertex =" << sumVertex << " numVertices=" << numVertices << std::endl;
            exit(1);
        }

        // calculate numElementsPU from VerticesPU/////////////////////////////
        unsigned int numElementsPU[PUNUM];
        for (int i = 0; i < PUNUM; i++) {
            numEdgesPU[i] = numEdges;
            numElementsPU[i] = numEdges * numVerticesPU[i];
        }

        // read in weights///////////////////////////////////////////
        for (int i = 0; i < 4 * PUNUM; i++) {
            weightDense[i] = aligned_alloc<float>(EXT_MEM_SZ);
        }
        readInWeight<PUNUM>(indicefstream, dataType, numElementsPU, weightDense);

        std::cout << "INFO: numVertice=" << numVertices << std::endl;
        std::cout << "INFO: numEdges=" << numEdges << std::endl;
    }

#ifdef __GRAPH_DEBUG__
#endif
}

template <int PUNUM>
int checkData(std::string goldenFile, ap_uint<32>* kernelID, float* kernelSimilarity) {
    int err = 0;
    char line[1024] = {0};
    std::fstream goldenfstream(goldenFile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Err : " << goldenFile << " file doesn't exist !" << std::endl;
    }

    std::unordered_map<int32_t, float> ref_map;
    int golden_num = 0;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        std::string tmp;
        int tmp_id;
        float tmp_data;

        data >> tmp_id;
        data >> tmp;
        tmp_data = std::stof(tmp);
        ref_map.insert(std::make_pair(tmp_id, tmp_data));
        golden_num++;
    }

    for (int i = 0; i < golden_num; i++) {
        std::cout << "kernel result: ID=" << kernelID[i] << " similarity=" << kernelSimilarity[i] << std::endl;
    }

    int index = 0;
    while (index < golden_num) {
        auto it = ref_map.find((int32_t)kernelID[index]);
        if (it != ref_map.end()) {
            float ref_result = it->second;
            if (std::abs(kernelSimilarity[index] - ref_result) > 0.000001) {
                std::cout << "Err: id=" << kernelID[index] << " golden_similarity=" << ref_result
                          << " kernel_similarity=" << kernelSimilarity[index] << std::endl;
                err++;
            }
        } else {
            std::cout << "not find! id=" << kernelID[index] << " kernel_similarity=" << kernelSimilarity[index]
                      << std::endl;
            err++;
        }
        index++;
    }

    return err;
}

#endif //#ifndef VT_GRAPH_UTILS_H
