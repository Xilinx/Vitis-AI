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

/*
 * usage: ./gemm_test.exe PATH_TO_XCLBIN/gemx.xclbin PATH_TO_XCLBIN/config_info.dat
 *
 */

#include <iomanip>
#include <cmath>
#include "xf_blas.hpp"

#define IDX2R(i, j, ld) (((i) * (ld)) + (j))
#define m 64 // a - mxk matrix
#define n 64 // b - kxn matrix
#define k 64 // c - mxn matrix

using namespace std;

BLAS_dataType* getGoldenMat(BLAS_dataType* a, BLAS_dataType* b, BLAS_dataType* c) {
    BLAS_dataType* goldenC;
    goldenC = (BLAS_dataType*)malloc(m * n * sizeof(BLAS_dataType));
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            BLAS_dataType l_val = 0;
            for (int i = 0; i < k; i++) {
                l_val += a[IDX2R(row, i, k)] * b[IDX2R(i, col, n)];
            }
            goldenC[IDX2R(row, col, n)] = l_val + c[IDX2R(row, col, n)];
        }
    }
    return goldenC;
}

bool compareGemm(BLAS_dataType* c, BLAS_dataType* goldenC, float p_TolRel = 1e-3, float p_TolAbs = 1e-5) {
    bool l_check = true;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            BLAS_dataType l_ref = goldenC[IDX2R(row, col, n)];
            BLAS_dataType l_result = c[IDX2R(row, col, n)];
            float l_diffAbs = abs(l_ref - l_result);
            float l_diffRel = l_diffAbs;
            if (goldenC[IDX2R(row, col, n)] != 0) {
                l_diffRel /= abs(l_ref);
            }
            bool check = (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
            if (!check) {
                cout << "golden result " << setprecision(10) << goldenC[IDX2R(row, col, n)]
                     << " is not equal to fpga result " << setprecision(10) << c[IDX2R(row, col, n)] << "\n";
                l_check = false;
            }
        }
    }
    return l_check;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << " usage: \n"
             << " gemm_test.exe gemx.xclbin config_info.dat 1\n"
             << " gemm_test.exe gemx.xclbin config_info.dat\n";
        return EXIT_FAILURE;
    }
    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
    string l_configFile(argv[l_argIdx++]);
    int l_numKernel = 1;

    if (argc == 4) {
        cout << "read custom number of kernels\n";
        l_numKernel = stoi(argv[l_argIdx++]);
    }

    xfblasEngine_t engineName = XFBLAS_ENGINE_GEMM;
    xfblasStatus_t status = xfblasCreate(l_xclbinFile.c_str(), l_configFile, engineName, l_numKernel);
    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Create Handle failed with error code: " << status << "\n";
        return EXIT_FAILURE;
    }

    int i, j; // i-row l_numKernel -1 ,j- column l_numKernel -1
    BLAS_dataType *a, *b, *c;

    posix_memalign((void**)&a, 4096, m * k * sizeof(BLAS_dataType));
    posix_memalign((void**)&b, 4096, k * n * sizeof(BLAS_dataType));
    posix_memalign((void**)&c, 4096, m * n * sizeof(BLAS_dataType));

    int ind = 1;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            a[IDX2R(i, j, k)] = ind;
        }
    }
    ind = 1;
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
            b[IDX2R(i, j, n)] = ind;
        }
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            c[IDX2R(i, j, n)] = 0;
        }
    }

    BLAS_dataType* goldenC = getGoldenMat(a, b, c);

    status = xfblasMallocRestricted(m, k, sizeof(*a), a, k, l_numKernel - 1);
    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Malloc memory for matrix A failed with error code: " << status << "\n";
        return EXIT_FAILURE;
    }

    status = xfblasMallocRestricted(k, n, sizeof(*b), b, n, l_numKernel - 1);

    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Malloc memory for matrix B failed with error code: " << status << "\n";
        xfblasDestroy();
        return EXIT_FAILURE;
    }

    status = xfblasMallocRestricted(m, n, sizeof(*c), c, n, l_numKernel - 1);

    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Malloc memory for matrix C failed with error code: " << status << "\n";
        xfblasDestroy();
        return EXIT_FAILURE;
    }

    status = xfblasSetMatrixRestricted(a, l_numKernel - 1);
    status = xfblasSetMatrixRestricted(b, l_numKernel - 1);
    status = xfblasSetMatrixRestricted(c, l_numKernel - 1);
    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Set Matrix failed with error code: " << status << "\n";
        xfblasDestroy();
        return EXIT_FAILURE;
    }

    status = xfblasGemm(XFBLAS_OP_N, XFBLAS_OP_N, m, n, k, 1, a, k, b, n, 1, c, n, l_numKernel - 1);

    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Matrix Multiplication failed with error code: " << status << "\n";
        xfblasDestroy();
        return EXIT_FAILURE;
    }

    status = xfblasGetMatrixRestricted(c, l_numKernel - 1);

    if (status != XFBLAS_STATUS_SUCCESS) {
        cout << "Get Matrix failed with error code: " << status << "\n";
        xfblasDestroy();
        return EXIT_FAILURE;
    }

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 10; j++) {
            cout << (c[IDX2R(i, j, k)]) << " ";
        }
        cout << "\n";
    }

    if (compareGemm(c, goldenC)) {
        cout << "Test passed!\n";
    } else {
        cout << "Test failed!\n";
    }

    xfblasFree(a, l_numKernel - 1);
    xfblasFree(b, l_numKernel - 1);
    xfblasFree(c, l_numKernel - 1);
    free(a);
    free(b);
    free(c);
    free(goldenC);

    xfblasDestroy(l_numKernel);

    return EXIT_SUCCESS;
}
