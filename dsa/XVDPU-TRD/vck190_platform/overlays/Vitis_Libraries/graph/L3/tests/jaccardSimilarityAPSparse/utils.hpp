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

template <typename MType>
union f_cast;

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
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
int checkData(std::string goldenFile, uint32_t* kernelID, float* kernelSimilarity) {
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
            for (int j = 0; j < length[i]; ++j) {
                buffer[i][j] = 1;
            }
        }
    } else {
        while (fstream.getline(line, sizeof(line))) {
            std::stringstream data(line);
            data >> buffer[row][id];
            id++;
            if (id >= length[row]) {
                id = 0;
                row++;
            }
        }
    }
}

template <int splitNm>
void readInWeight(std::fstream& fstream,    // input: file stream
                  int dataType,             // 0:int32, 1:float
                  uint32_t length[splitNm], // input: value numbers of each buffer
                  float** buffer) {         // output: output buffers
    int id = 0;
    int row = 0;
    char line[1024] = {0};
    if (!fstream) {
        for (int i = 0; i < 4 * splitNm; ++i) {
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
                f_cast<float> tmp0;
                tmp0.i = tmp;
                buffer[row][id] = tmp0.f;
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
                        unsigned int length[PUNUM], // input: value numbers of each buffer
                        uint32_t* buffer1[PUNUM],   // output: first array
                        float* buffer2[PUNUM]) {    // output: second array
    int id = 0;
    int row = 0;
    std::string line;
    bool flag = 0;
    bool flag1 = 0;
    f_cast<float> tmp0;
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
            if (dataType == 0) {
                tmp0.i = 1;
                buffer2[row][id] = tmp0.f;
            } else
                buffer2[row][id] = 1.0;
        } else {
            float tmp;
            data >> tmp;
            if (dataType == 0) {
                tmp0.i = (int32_t)tmp;
                buffer2[row][id] = tmp0.f;
            } else
                buffer2[row][id] = tmp;
        }
        id++;
        if (id >= length[row]) {
            id = 0;
            row++;
        }
    }
}
