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

#ifndef GEMM_HELPER_HPP
#define GEMM_HELPER_HPP

#include <cmath>
#include <iomanip>
#include <string>

#define IDX2R(i, j, ld) (((i) * (ld)) + (j))

using namespace std;

// Deprecated (Recommend using gemm_mkl to generate the golden output.)
BLAS_dataType* getGoldenMat(BLAS_dataType* a, BLAS_dataType* b, BLAS_dataType* c, int m, int k, int n) {
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

bool compareGemm(BLAS_dataType* c, BLAS_dataType* goldenC, int m, int n, float p_TolRel = 1e-3, float p_TolAbs = 1e-5) {
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
                cout << "#(" << row << ", " << col << ") golden result" << setprecision(10)
                     << goldenC[IDX2R(row, col, n)] << " is not equal to fpga result " << setprecision(10)
                     << c[IDX2R(row, col, n)] << "\n";
                l_check = false;
            }
        }
    }
    return l_check;
}

#endif
