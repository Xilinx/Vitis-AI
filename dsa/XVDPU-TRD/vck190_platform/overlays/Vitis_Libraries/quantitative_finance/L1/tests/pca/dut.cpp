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

/**
 * @file dut.cpp
 *
 * @brief This file contains top function of the test case.
 */

#include "pca.hpp"

#define TEST_DT double
#define TEST_MAXVARS 15
#define TEST_MAXOBS 80
#define N_FACTORS 3
#define PCA_NCU 1

extern "C" void dut(unsigned rows,
                    unsigned cols,
                    TEST_DT input[TEST_MAXVARS][TEST_MAXOBS],
                    TEST_DT outputLoadings[TEST_MAXVARS][N_FACTORS]) {
#pragma HLS ARRAY_PARTITION variable = outputLoadings dim = 2 complete
    xf::fintech::PCA<TEST_DT, N_FACTORS, PCA_NCU, TEST_MAXVARS, TEST_MAXOBS> pca(rows, cols, input);
    pca.getLoadingsMatrix(outputLoadings);
}
