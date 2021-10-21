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

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>

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

template <int splitNm>
int checkData(std::string goldenFile, int32_t* kernelID, float* kernelSimilarity) {
    int err = 0;
    char line[1024] = {0};
    std::fstream goldenfstream(goldenFile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error: " << goldenFile << " file doesn't exist !" << std::endl;
        exit(1);
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

    int index = 0;
    while (index < golden_num) {
        auto it = ref_map.find((int32_t)kernelID[index]);
        if (it != ref_map.end()) {
            float ref_result = it->second;
            if (std::abs(kernelSimilarity[index] - ref_result) > 0.000001) {
                std::cout << "Error: id=" << kernelID[index] << " golden_similarity=" << ref_result
                          << " kernel_similarity=" << kernelSimilarity[index] << std::endl;
                err++;
            }
        } else {
            std::cout << "Error: not find! id=" << kernelID[index] << " kernel_similarity=" << kernelSimilarity[index]
                      << std::endl;
            err++;
        }
        index++;
    }

    return err;
}

template <int splitNm>
void readInWeight(std::fstream& fstream,          // input: file stream
                  int32_t numEdges,               // number edges
                  int32_t numVerticesPU[splitNm], // input: value numbers of each buffer
                  int32_t** buffer) {             // output: output buffers
    int id = 0;
    int counter = 0;
    int tmpID[4 * splitNm] = {0};
    int row = 0;
    int splitID = 0;
    char line[1024] = {0};
    int32_t edgeAlign8 = ((numEdges + 7) / 8) * 8;
    int32_t VidChannel = (numVerticesPU[0] + 3) / 4;
    int32_t prev = 0;
    while (fstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        // float tmp;
        int32_t tmp;
        data >> tmp;
        buffer[row][tmpID[row] * edgeAlign8 + id] = tmp;
        id++;
        counter++;
        if (counter == (numVerticesPU[splitID] * numEdges)) {
            splitID++;
            counter = 0;
            if (splitID < splitNm) {
                VidChannel = (numVerticesPU[splitID] + 3) / 4;
            }
            id = 0;
            row++;
        } else if ((tmpID[row] == (VidChannel - 1)) && (id >= numEdges)) {
            id = 0;
            row++;
        } else if (id >= numEdges) {
            id = 0;
            tmpID[row]++;
        }
    }
    fstream.close();
}

template <int splitNm>
void generateSourceParams(int32_t numVerticesPU[splitNm],
                          int32_t numEdges,
                          int sourceID,
                          int** weightsDense,
                          int32_t& sourceNUM,
                          int32_t** sourceWeight) {
    sourceNUM = (int32_t)((numEdges + 7) / 8) * 8;
    *sourceWeight = aligned_alloc<int32_t>(sourceNUM);

    int32_t id = 0;
    int32_t row = 0;
    int32_t offset[splitNm + 1];
    offset[0] = 0;
    for (int i = 0; i < splitNm; i++) {
        offset[i + 1] = numVerticesPU[i] + offset[i];
        if ((sourceID >= offset[i]) && (sourceID < offset[i + 1])) {
            id = i;
            row = sourceID - offset[i];
        }
    }
    for (int i = 0; i < sourceNUM; i++) {
        sourceWeight[0][i] = weightsDense[id * 4 + row % 4][(row / 4) * sourceNUM + i];
    }
}
