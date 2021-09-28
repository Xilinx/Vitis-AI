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
#include <cstdlib>
#include "utils.hpp"
#include "uut_top.hpp"
#include "binFiles.hpp"

using namespace std;

int main(int argc, char** argv) {
    string data_dir = argv[1];
    vector<BLAS_dataType> l_A, l_B, l_C, l_ref;
    vector<BLAS_dataType> l_R(BLAS_matrixSizeC, 0);

    readBin(data_dir + "matA.bin", sizeof(BLAS_dataType) * BLAS_m * BLAS_k, l_A);
    readBin(data_dir + "matB.bin", sizeof(BLAS_dataType) * BLAS_k * BLAS_n, l_B);
    readBin(data_dir + "matC.bin", sizeof(BLAS_dataType) * BLAS_m * BLAS_n, l_C);
    readBin(data_dir + "golden.bin", sizeof(BLAS_dataType) * BLAS_m * BLAS_n, l_ref);

    uut_top(BLAS_m, BLAS_n, BLAS_k, BLAS_alpha, BLAS_beta, l_A.data(), l_B.data(), l_C.data(), l_R.data());

    int err = 0;
    bool l_compare = compare(BLAS_matrixSizeC, l_R.data(), l_ref.data(), err);

    if (l_compare) {
        cout << "Pass!\n";
        return 0;
    } else {
        cout << "Fail with " << err << " errors!\n";
        return -1;
    }
}
