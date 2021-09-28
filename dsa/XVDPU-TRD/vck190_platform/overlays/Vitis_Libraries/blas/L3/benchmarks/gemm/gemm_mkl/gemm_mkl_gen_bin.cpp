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

#include "gemm_mkl_helper.hpp"

#include <fstream>
#include <string>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: gemm_mkl m k n dir\n");
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    XFBLAS_dataType *a, *b, *c, alpha = 1., beta = 1.;

    // Generating Random Input
    a = createMat(m, k);
    b = createMat(k, n);
    c = createMat(m, n);

    ofstream outFile;
    string data_dir(argv[4]);
#ifdef USE_SHORT
    short *a_short, *b_short, *c_short;
    if (posix_memalign((void**)&a_short, 4096, (size_t)m * (size_t)k * sizeof(short)) != 0) {
        printf("[ERROR] failed to create the matrix a_short\n");
        exit(1);
    }
    if (posix_memalign((void**)&b_short, 4096, (size_t)k * (size_t)n * sizeof(short)) != 0) {
        printf("[ERROR] failed to create the matrix a_short\n");
        exit(1);
    }
    if (posix_memalign((void**)&c_short, 4096, (size_t)m * (size_t)n * sizeof(short)) != 0) {
        printf("[ERROR] failed to create the matrix a_short\n");
        exit(1);
    }
    printf("[WARNING] The short data is currently casted from float datatype.\n");
    for (int i = 0; i < m * k; i++) a_short[i] = (short)a[i];
    for (int i = 0; i < k * n; i++) b_short[i] = (short)b[i];
    for (int i = 0; i < m * n; i++) c_short[i] = (short)c[i];

    outFile.open(data_dir + "matA_in_" + to_string(m) + "_" + to_string(k) + ".bin", ofstream::binary);
    outFile.write((char*)a_short, sizeof(short) * m * k);
    outFile.close();

    outFile.open(data_dir + "matB_in_" + to_string(k) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)b_short, sizeof(short) * k * n);
    outFile.close();

    outFile.open(data_dir + "matC_in_" + to_string(m) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)c_short, sizeof(short) * m * n);
    outFile.close();
#else
    outFile.open(data_dir + "matA_in_" + to_string(m) + "_" + to_string(k) + ".bin", ofstream::binary);
    outFile.write((char*)a, sizeof(XFBLAS_dataType) * m * k);
    outFile.close();

    outFile.open(data_dir + "matB_in_" + to_string(k) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)b, sizeof(XFBLAS_dataType) * k * n);
    outFile.close();

    outFile.open(data_dir + "matC_in_" + to_string(m) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)c, sizeof(XFBLAS_dataType) * m * n);
    outFile.close();
#endif

    // Generating Golden Output
    GEMM_MKL(m, n, k, alpha, beta, a, b, c);

#ifdef USE_SHORT
    for (int i = 0; i < m * n; i++) c_short[i] = (short)c[i];

    outFile.open(data_dir + "matC_out_" + to_string(m) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)c_short, sizeof(short) * m * n);
    outFile.close();

    free(a_short);
    free(b_short);
    free(c_short);
#else
    outFile.open(data_dir + "matC_out_" + to_string(m) + "_" + to_string(n) + ".bin", ofstream::binary);
    outFile.write((char*)c, sizeof(XFBLAS_dataType) * m * n);
    outFile.close();
#endif
    free(a);
    free(b);
    free(c);

    return 0;
}
