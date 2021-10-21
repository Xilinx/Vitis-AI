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
 * @file quad_engine .cpp
 * @brief HLS implementation of the numerical integration engines
 */

#include <ap_fixed.h>
#include <hls_stream.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "hls_math.h"

#define TEST_DT float
#define MAX_NUMBER_TESTS 1024

// the function to be integrated
TEST_DT polynomial(TEST_DT x, void* p) {
    return (0.1 * x * x * x) + (4 * x * x) + (15.5 * x) + 33.3;
}

#define MAX_ITERATIONS 10000
#define MAX_DEPTH 20
#define XF_INTEGRAND_FN polynomial
#define XF_USER_DATA_TYPE void
#include "xf_fintech/quadrature.hpp"

extern "C" {

/// @brief Kernel top level
///
/// This is the top level kernel and represents the interface presented to the
/// host.
///
/// @param[in]  a_in      Input parameters read as a vector bus type
/// @param[in]  b_in      Input parameters read as a vector bus type
/// @param[in]  method_in Input parameters read as a vector bus type
/// @param[in]  tol_in    Input parameters read as a vector bus type
/// @param[in]  num      Total number of input data sets to process
/// @param[out] res      Output parameters read as a vector bus type

void quad_kernel(TEST_DT* a, TEST_DT* b, TEST_DT* method, TEST_DT* tol, int num, TEST_DT* res) {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = in0_port
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = in1_port
#pragma HLS INTERFACE m_axi port = method offset = slave bundle = in2_port
#pragma HLS INTERFACE m_axi port = tol offset = slave bundle = in3_port
#pragma HLS INTERFACE m_axi port = res offset = slave bundle = out0_port

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = method bundle = control
#pragma HLS INTERFACE s_axilite port = tol bundle = control
#pragma HLS INTERFACE s_axilite port = res bundle = control

#pragma HLS INTERFACE s_axilite port = num bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    TEST_DT local_a[MAX_NUMBER_TESTS];
    TEST_DT local_b[MAX_NUMBER_TESTS];
    TEST_DT local_method[MAX_NUMBER_TESTS];
    TEST_DT local_tol[MAX_NUMBER_TESTS];
    TEST_DT local_res[MAX_NUMBER_TESTS];

    for (int i = 0; i < num; i++) {
        local_a[i] = a[i];
        local_b[i] = b[i];
        local_method[i] = method[i];
        local_tol[i] = tol[i];
    }

    for (int i = 0; i < num; i++) {
        if (local_method[i] == 0) {
            xf::fintech::trap_integrate<TEST_DT>(local_a[i], local_b[i], local_tol[i], &local_res[i], NULL);
        } else if (local_method[i] == 1) {
            xf::fintech::simp_integrate<TEST_DT>(local_a[i], local_b[i], local_tol[i], &local_res[i], NULL);
        } else if (local_method[i] == 2) {
            xf::fintech::romberg_integrate<TEST_DT>(local_a[i], local_b[i], local_tol[i], &local_res[i], NULL);
        }
    }

    for (int i = 0; i < num; i++) {
        res[i] = local_res[i];
    }
}
} // extern C
