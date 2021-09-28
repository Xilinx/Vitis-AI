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

#ifndef __SYNTHESIS__
#include <algorithm>
#include <iostream>
#include <limits>
#include <string.h>
#include <sys/time.h>
#endif

#include <hls_stream.h>
#include <stdlib.h>
#ifdef _HLS_TEST_
#include "kernel_pseudosqrt.hpp"
#endif

#ifndef __SYNTHESIS__

union DTConvert {
    uint64_t fixed;
    DT fp;
};

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
}

// Compute time difference
unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

template <typename T>
void covCalc(int rows, int cols, T* inMat, T* outCov) {
    T* tmpMat = new T[rows * cols];
    for (int i = 0; i < rows; i++) {
        DT ave = 0.0;
        for (int j = 0; j < cols; j++) {
            ave += inMat[i * cols + j] / cols;
        }
        for (int j = 0; j < cols; j++) {
            tmpMat[i * cols + j] = inMat[i * cols + j] - ave;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j <= i; j++) {
            DT tmp = 0.0;
            for (int k = 0; k < cols; k++) {
                tmp += tmpMat[i * cols + k] * tmpMat[j * cols + k];
            }
            outCov[i * cols + j] = tmp / (cols - 1);
            outCov[j * cols + i] = tmp / (cols - 1);
        }
    }
    delete[] tmpMat;
}

// generate SPD matrix
void genSPD(int nrows, DT* matOrigin, DT* matrix) {
#ifdef _DEBUG_SOLVER_
    std::cout << "-------------------- SPD matrix----------------- \n";
#endif
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < nrows; ++j) {
            DT sum = 0.0;
            for (int k = 0; k < nrows; ++k) {
                sum += matOrigin[i * nrows + k] * matOrigin[j * nrows + k];
            }
            matrix[i * nrows + j] = sum;
#ifdef _DEBUG_SOLVER_
            if ((i < 4) && (j < 4)) {
                std::cout << matrix[i * nrows + j] << "\t";
            }
#endif
        }
#ifdef _DEBUG_SOLVER_
        if (i < 4) {
            std::cout << "\n";
        }
#endif
    }
#ifdef _DEBUG_SOLVER_
    std::cout << "--------------------------------------------------- \n";
#endif
}

// generate SPD matrix
DT norm(int nrows, DT* matOrigin, DT* result) {
    DT sum = 0.0;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < nrows; ++j) {
            sum +=
                (matOrigin[i * nrows + j] - result[i * nrows + j]) * (matOrigin[i * nrows + j] - result[i * nrows + j]);
#ifdef _DEBUG_SOLVER_
            std::cout << "befor = " << matOrigin[i * nrows + j] << "\t after = " << result[i * nrows + j] << std::endl;
#endif
        }
    }
    return std::sqrt(sum);
}

// Arguments parser
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
    // Initialize parserl
    ArgParser parser(argc, argv);

    // Initialize paths addresses
    std::string xclbin_path;
    std::string num_str;
    int nrows;

    // Read In paths addresses
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "INFO: input path is not set!\n";
    }
    if (!parser.getCmdOption("-nrows", num_str)) {
        nrows = 2;
        std::cout << "INFO: number of rows/column is not set!\n";
    } else {
        nrows = std::stoi(num_str);
    }

    DT tolerance = (sizeof(DT) == 4) ? 1e-1 : 1e-8;
    DT* matOrigin = new DT[nrows * nrows];
    DT* matrix = aligned_alloc<DT>(nrows * nrows);
    DT* golden = new DT[nrows * nrows];
    DT* result = aligned_alloc<DT>(nrows * nrows);
    // Generate random matrix
    int seed = (unsigned)time(NULL);
    seed = 55;
    std::cout << "INFO: seed = " << seed << std::endl;
    srand(seed);
#ifdef _DEBUG_SOLVER_
    std::cout << "-------------------- Original matrix----------------- \n";
#endif
    for (int k = 0; k < nrows; ++k) {
        for (int j = 0; j < nrows; ++j) {
            matOrigin[k * nrows + j] = 100 * (-0.5 + rand() / DT(RAND_MAX));
#ifdef _DEBUG_SOLVER_
            if ((k < 4) && (j < 4)) {
                std::cout << matOrigin[k * nrows + j] << "\t";
            }
#endif
        }
#ifdef _DEBUG_SOLVER_
        if (k < 4) {
            std::cout << std::endl;
        }
#endif
    }
#ifdef _DEBUG_SOLVER_
    std::cout << "--------------------------------------------------- \n";
#endif
    covCalc<DT>(nrows, nrows, matOrigin, matrix);
    covCalc<DT>(nrows, nrows, matOrigin, golden);

#ifdef _DEBUG_SOLVER_
    std::cout << "-------------------- Original matrix----------------- \n";
    for (int k = 0; k < nrows; ++k) {
        for (int j = 0; j < nrows; ++j) {
            if ((k < 4) && (j < 4)) {
                std::cout << matrix[k * nrows + j] << "\t";
            }
        }
        if (k < 4) {
            std::cout << std::endl;
        }
    }
    std::cout << "--------------------------------------------------- \n";
#endif

#ifdef _USE_STRM_
    hls::stream<ap_uint<DTLen * TO> > inMat;
    hls::stream<ap_uint<DTLen * TO> > outMat;
    int size0 = nrows * nrows;
    int size = (nrows * nrows + TO - 1) / TO;
    for (int i = 0; i < size; ++i) {
        ap_uint<DTLen* TO> tmp = 0;
        for (int j = 0; j < TO; ++j) {
            int index = i * TO + j;
            if (index < size0) {
                DTConvert tmp1;
                tmp1.fp = matrix[index];
                tmp.range(DTLen * (j + 1) - 1, DTLen * j) = tmp1.fixed;
            }
        }
        inMat.write(tmp);
    }
#endif
#ifdef _HLS_TEST_

#ifdef _USE_STRM_
    kernel_pseudosqrt_0(nrows, inMat, outMat);
#else
    kernel_pseudosqrt_0(nrows, matrix, result);
#endif
#endif

#ifdef _USE_STRM_
    for (int i = 0; i < size; ++i) {
        ap_uint<DTLen* TO> tmp = outMat.read();
        for (int j = 0; j < TO; ++j) {
            int index = i * TO + j;
            if (index < size0) {
                DTConvert tmp1;
                tmp1.fixed = tmp.range(DTLen * (j + 1) - 1, DTLen * j);
                result[index] = tmp1.fp;
            }
        }
    }
#endif

#ifdef _DEBUG_SOLVER_
    std::cout << "-------------------- Output matrix----------------- \n";
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < nrows; ++j) {
            if ((i < 4) && (j < 4)) {
                std::cout << result[i * nrows + j] << "\t";
            }
        }
        if (i < 4) {
            std::cout << "\n";
        }
    }
    std::cout << "--------------------------------------------------- \n";
#endif
    DT* matCarre = new DT[nrows * nrows];
    genSPD(nrows, result, matCarre);

    DT error = norm(nrows, golden, matCarre);
    std::cout << "INFO: Residual = " << error << std::endl;

    delete[] matOrigin;
    delete[] matCarre;
    delete[] golden;
    free(matrix);
    free(result);

    if ((error > tolerance) || (error < 0)) {
        std::cout << "INFO: Results are fault !" << std::endl;
        return 1;
    } else {
        std::cout << "INFO: Results are correct !" << std::endl;
        return 0;
    }
}
#endif
