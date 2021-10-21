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

#ifndef XF_BLAS_HELPER_HPP
#define XF_BLAS_HELPER_HPP

#include <fstream>
#include <string>
#include <unordered_map>

using namespace std;

namespace xf {

namespace blas {

#ifndef BLAS_streamingKernel
typedef enum { OpControl, OpGemv, OpGemm, OpTransp, OpResult, OpFail, OpFcn } OpType;
#endif

class BLASArgs {
   public:
    virtual ~BLASArgs() {}
    virtual size_t sizeInBytes() = 0;
    virtual char* asByteArray() = 0;
};

xfblasStatus_t buildConfigDict(string p_configFile,
                               xfblasEngine_t p_engineName,
                               unordered_map<string, string>* p_configDict) {
    unordered_map<string, string> l_configDict;
    ifstream l_configInfo(p_configFile);
    bool l_good = l_configInfo.good();
    if (!l_good) {
        return XFBLAS_STATUS_NOT_INITIALIZED;
    }
    if (l_configInfo.is_open()) {
        string line;
        string key;
        string value;
        string equalSign = "=";
        while (getline(l_configInfo, line)) {
            int index = line.find(equalSign);
            if (index == 0) continue;
            key = line.substr(0, index);
            value = line.substr(index + 1);
            l_configDict[key] = value;
        }
    }

    l_configInfo.close();
#if BLAS_streamingKernel
    // Additional limit for different engines
    if (p_engineName == XFBLAS_ENGINE_GEMM) {
        if (l_configDict.find("BLAS_mParWords") != l_configDict.end()) {
            int l_mBlock = stoi(l_configDict["BLAS_mParWords"]);
            int l_kBlock = stoi(l_configDict["BLAS_kParWords"]);
            int l_nBlock = stoi(l_configDict["BLAS_nParWords"]);
            int l_ddrWidth = stoi(l_configDict["BLAS_parEntries"]);
            int l_maxBlock = max(l_mBlock, max(l_kBlock, l_nBlock));
            int l_minSize = l_ddrWidth * l_maxBlock;
            l_configDict["minSize"] = to_string(l_minSize);
        } else {
            return XFBLAS_STATUS_NOT_INITIALIZED;
        }
    }
#else
    // Additional limit for different engines
    if (p_engineName == XFBLAS_ENGINE_GEMM) {
        if (l_configDict.find("BLAS_gemmMBlocks") != l_configDict.end()) {
            int l_mBlock = stoi(l_configDict["BLAS_gemmMBlocks"]);
            int l_kBlock = stoi(l_configDict["BLAS_gemmKBlocks"]);
            int l_nBlock = stoi(l_configDict["BLAS_gemmNBlocks"]);
            int l_ddrWidth = stoi(l_configDict["BLAS_ddrWidth"]);
            int l_maxBlock = max(l_mBlock, max(l_kBlock, l_nBlock));
            int l_minSize = l_ddrWidth * l_maxBlock;
            l_configDict["minSize"] = to_string(l_minSize);
        } else {
            return XFBLAS_STATUS_NOT_INITIALIZED;
        }
    }
#endif
    *p_configDict = l_configDict;
    return XFBLAS_STATUS_SUCCESS;
}

int getPaddedSize(int p_size, int p_minSize) {
    return p_size + p_minSize - 1 - (p_size - 1) % p_minSize;
}

int getTypeSize(string p_typeName) {
    if (p_typeName == "float") {
        return sizeof(float);
    } else if (p_typeName == "short") {
        return sizeof(short);
    } else if (p_typeName == "int") {
        return sizeof(int);
    } else {
        return 0;
    }
}

} // namespace blas

} // namespace xf

#endif
