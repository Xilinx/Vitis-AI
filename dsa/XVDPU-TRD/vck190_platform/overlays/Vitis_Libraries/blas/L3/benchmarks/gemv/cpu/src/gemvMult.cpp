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

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <mkl.h>

#include "utils.hpp"

using namespace std;

typedef chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

void readBin(string name, char* mat, unsigned int totalSize) {
    ifstream inFile;
    inFile.open(name, ifstream::binary);
    if (inFile.is_open()) {
        inFile.read((char*)mat, totalSize);
        inFile.close();
    } else {
        cerr << "Could not find " << name << endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <Matrix Row> <Matrix Col> <Matrix Path> iterations" << endl;
        return EXIT_FAILURE;
    }
    mkl_set_num_threads(32);

    int p_m = atoi(argv[1]);

    int p_n = atoi(argv[2]);

    int matrixSize = p_m * p_n;

    BLAS_dataType* h_A = (BLAS_dataType*)malloc(matrixSize * sizeof(BLAS_dataType));
    BLAS_dataType* h_x = (BLAS_dataType*)malloc(p_n * sizeof(BLAS_dataType));
    BLAS_dataType* h_b = (BLAS_dataType*)malloc(p_m * sizeof(BLAS_dataType));
    BLAS_dataType* h_r = (BLAS_dataType*)malloc(p_m * sizeof(BLAS_dataType));

    string filepath = argv[3];
    readBin(filepath + "A.mat", (char*)h_A, matrixSize * sizeof(BLAS_dataType));
    readBin(filepath + "x.mat", (char*)h_x, p_n * sizeof(BLAS_dataType));
    readBin(filepath + "b.mat", (char*)h_b, p_m * sizeof(BLAS_dataType));

    int iterations = atoi(argv[4]);

    TimePointType l_tp[4];
    l_tp[0] = chrono::high_resolution_clock::now();

#if USE_DOUBLE_PRECISION
    cblas_dgemv(CblasRowMajor, CblasNoTrans, p_m, p_n, 1, h_A, p_n, h_x, 1, 1, h_r, 1);
#else
    cblas_sgemv(CblasRowMajor, CblasNoTrans, p_m, p_n, 1, h_A, p_n, h_x, 1, 1, h_r, 1);
#endif
    l_tp[1] = chrono::high_resolution_clock::now();

    int err = 0;

    compare(p_m, h_b, h_r, err);

    l_tp[2] = chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
#if USE_DOUBLE_PRECISION
        cblas_dgemv(CblasRowMajor, CblasNoTrans, p_m, p_n, 1, h_A, p_n, h_x, 1, 1, h_r, 1);
#else
        cblas_sgemv(CblasRowMajor, CblasNoTrans, p_m, p_n, 1, h_A, p_n, h_x, 1, 1, h_r, 1);
#endif
    }
    l_tp[3] = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed_cold = l_tp[1] - l_tp[0];
    chrono::duration<double> elapsed_hot = l_tp[3] - l_tp[2];
    double duration_cold = elapsed_cold.count();
    double duration_hot = elapsed_hot.count();

    cout << "Execution time is (cold)" << duration_cold << "s." << endl;
    cout << "Execution time is (hot)" << duration_hot / iterations << "s." << endl;

    free(h_A);
    free(h_x);
    free(h_b);
    free(h_r);

    if (err == 0) {
        cout << "Results verified." << endl;
        return EXIT_SUCCESS;
    } else {
        cout << "There are in total " << err << " mismatches in the solution." << endl;
        return EXIT_FAILURE;
    }
}
