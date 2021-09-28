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
 * @brief This file contains top function of test case.
 */

#include "dut.hpp"
#include <ap_int.h>
#include "xf_fintech/rng.hpp"

extern "C" void dut(double input[ASSETS * (2 * ASSETS + 1)],
                    double output[ASSETS * (2 * ASSETS - 1)],
                    int func,
                    int calls) {
    // func = 0 return mean
    // func = 1 return variance
    // func = 1 return correlation

    double input_ltm[2 * ASSETS + 1][ASSETS];
    for (int i = 0; i < 2 * ASSETS + 1; i++) {
        for (int j = 0; j < ASSETS; j++) {
            input_ltm[i][j] = input[i * ASSETS + j];
        }
    }

    xf::fintech::MultiVariateNormalRng<double, ASSETS, BUFFDEPTH> mrng;
    mrng.init(12, input_ltm);

    if (func == 0) {
        double tmp[ASSETS * 2];
        for (int i = 0; i < ASSETS * 2; i++) {
#pragma HLS pipeline
            tmp[i] = 0;
        }
    FUNC_0_LOOP:
        for (int t = 0; t < calls; t++) {
            for (int i = 0; i < BUFFDEPTH; i++) {
            LOOP_FUNC_0:
                for (int j = 0; j < ASSETS; j++) {
#pragma HLS pipeline
                    double local_result[2];
                    // mrng.next(local_result[0], local_result[1]);
                    tmp[j * 2] += local_result[0];
                    tmp[j * 2 + 1] += local_result[1];
                }
            }
        }
        for (int i = 0; i < ASSETS * 2; i++) {
#pragma HLS pipeline
            output[i] = tmp[i] / (calls * BUFFDEPTH);
        }
    } else if (func == 1) {
        double tmp[ASSETS * 2];
        for (int i = 0; i < ASSETS * 2; i++) {
#pragma HLS pipeline
            tmp[i] = 0;
        }
    FUNC_1_LOOP:
        for (int t = 0; t < calls; t++) {
            for (int i = 0; i < BUFFDEPTH; i++) {
            LOOP_FUNC_1:
                for (int j = 0; j < ASSETS; j++) {
#pragma HLS pipeline
                    double local_result[2];
                    mrng.next(local_result[0], local_result[1]);
                    tmp[j * 2] += local_result[0] * local_result[0];
                    tmp[j * 2 + 1] += local_result[1] * local_result[1];
                }
            }
        }
        for (int i = 0; i < ASSETS * 2; i++) {
#pragma HLS pipeline
            output[i] = tmp[i] / (calls * BUFFDEPTH);
        }
    } else if (func == 2) {
        double local_result[(ASSETS * 2 - 1) * ASSETS];
        for (int i = 0; i < (ASSETS * 2 - 1) * ASSETS; i++) {
#pragma HLS pipeline
            local_result[i] = 0;
        }
    FUNC_2_LOOP:
        for (int t = 0; t < calls; t++) {
            double tmp[ASSETS * 2][BUFFDEPTH];
        LOOP_FUNC_2_1:
            for (int i = 0; i < BUFFDEPTH; i++) {
                for (int j = 0; j < ASSETS; j++) {
#pragma HLS pipeline
                    // mrng.next(tmp[j * 2][i], tmp[j * 2 + 1][i]);
                    double out0, out1;
                    mrng.next(out0, out1);
                    tmp[j * 2][i] = out0;
                    tmp[j * 2 + 1][i] = out1;
                }
            }
        LOOP_FUNC_2_3:
            for (int i = 0; i < BUFFDEPTH; i++) {
                int cnt = 0;
                for (int m = 1; m < ASSETS * 2; m++) {
                    for (int n = 0; n < m; n++) {
#pragma HLS pipeline
                        local_result[cnt] += tmp[m][i] * tmp[n][i];
                        cnt++;
                    }
                }
            }
        }
        for (int i = 0; i < (ASSETS * 2 - 1) * ASSETS; i++) {
#pragma HLS pipeline
            output[i] = local_result[i] / (calls * BUFFDEPTH);
        }
    }
}
