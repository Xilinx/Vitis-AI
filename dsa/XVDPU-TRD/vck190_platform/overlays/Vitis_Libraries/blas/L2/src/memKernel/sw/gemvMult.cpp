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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <cassert>

// This file is required for OpenCL C++ wrapper APIs
#include "gemvMult.hpp"
#include "utils.hpp"
#include "binFiles.hpp"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        cout << "Usage: " << argv[0] << " <XCLBIN File> <Matrix Row> <Matrix Col> <Matrix Path> [device id]" << endl;
        return EXIT_FAILURE;
    }

    int32_t l_index = 0;

    string binaryFile = argv[++l_index];

    int p_m = atoi(argv[++l_index]);
    assert(p_m % BLAS_numChannels == 0);

    int p_n = atoi(argv[++l_index]);
    assert(p_n % BLAS_parEntries == 0);

    int matrixSize = p_m * p_n;

    // I/O Data Vectors
    host_buffer_t<BLAS_dataType> h_A(matrixSize);
    host_buffer_t<BLAS_dataType> h_x(p_n);
    host_buffer_t<BLAS_dataType> h_b(p_m);
    host_buffer_t<BLAS_dataType> h_r(p_m);

    string filepath = argv[++l_index];
    readBin(filepath + "A.mat", h_A.size() * sizeof(BLAS_dataType), h_A);
    readBin(filepath + "x.mat", h_x.size() * sizeof(BLAS_dataType), h_x);
    readBin(filepath + "b.mat", h_b.size() * sizeof(BLAS_dataType), h_b);

    int l_deviceId = 0;
    if (argc > l_index) l_deviceId = atoi(argv[++l_index]);

    FPGA fpga(l_deviceId);
    fpga.xclbin(binaryFile);
    GemvKernel<BLAS_dataType, BLAS_numChannels> gemv(p_m, p_n, &fpga);

    gemv.getCU("krnl_gemv");
    gemv.setMem(h_A, h_x, h_r);
    gemv.run();
    gemv.getMem();

    int err = 0;
    compare(p_m, h_b.data(), h_r.data(), err);
    if (err == 0) {
        cout << "Results verified." << endl;
        return EXIT_SUCCESS;
    } else {
        cout << "There are in total " << err << " mismatches in the solution." << endl;
        return EXIT_FAILURE;
    }
}
