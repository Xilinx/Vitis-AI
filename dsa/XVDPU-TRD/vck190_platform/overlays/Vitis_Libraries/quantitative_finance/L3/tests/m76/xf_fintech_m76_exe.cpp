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
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/
/*-- DISCLAIMER AND CRITICAL APPLICATIONS */
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/
/*-- */
/*-- (c) Copyright 2019 Xilinx, Inc. All rights reserved. */
/*-- */
/*-- This file contains confidential and proprietary information of Xilinx, Inc. and is protected under U.S. and */
/*-- international copyright and other intellectual property laws. */
/*-- */
/*-- DISCLAIMER */
/*-- This disclaimer is not a license and does not grant any rights to the materials distributed herewith. Except as */
/*-- otherwise provided in a valid license issued to you by Xilinx, and to the maximum extent permitted by applicable */
/*-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
 */
/*-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON- */
/*-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in contract or
 * tort,*/
/*-- including negligence, or under any other theory of liability) for any loss or damage of any kind or nature */
/*-- related to, arising under or in connection with these materials, including for any direct, or any indirect, */
/*-- special, incidental, or consequential loss or damage (including loss of data, profits, goodwill, or any type of */
/*-- loss or damage suffered as a retVal of any action brought by a third party) even if such damage or loss was */
/*-- reasonably foreseeable or Xilinx had been advised of the possibility of the same. */
/*-- */
/*-- CRITICAL APPLICATIONS */
/*-- Xilinx products are not designed or intended to be fail-safe, or for use in any application requiring fail-safe */
/*-- performance, such as life-support or safety devices or systems, Class III medical devices, nuclear facilities, */
/*-- applications related to the deployment of airbags, or any other applications that could lead to death, personal */
/*-- injury, or severe property or environmental damage (individually and collectively, "Critical */
/*-- Applications"). Customer assumes the sole risk and liability of any use of Xilinx products in Critical */
/*-- Applications, subject only to applicable laws and regulations governing limitations on product liability. */
/*-- */
/*-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES. */
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <vector>

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

struct test_data_type {
    float S;
    float K;
    float r;
    float sigma;
    float T;
    float lambda;
    float kappa;
    float delta;
    float exp;
};

struct test_data_type test_data[] = {
    {10.021600, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 1.969068},
    {10.021600, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 2.005603},
    {10.021600, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 2.106863},
    {10.021600, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 2.485681},
    {10.021600, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 3.811954},
    {10.021600, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 5.774945},
    {10.021600, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 8.821378},
    {19.081700, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 5.757761},
    {19.081700, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 5.837310},
    {19.081700, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 6.057548},
    {19.081700, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 6.846772},
    {19.081700, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 9.324169},
    {19.081700, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 12.625823},
    {19.081700, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 17.332820},
    {41.416200, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 19.067555},
    {41.416200, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 19.225610},
    {41.416200, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 19.660236},
    {41.416200, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 21.165239},
    {41.416200, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 25.625022},
    {41.416200, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 31.208559},
    {41.416200, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 38.733838},
    {80.435600, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 48.488604},
    {80.435600, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 48.713268},
    {80.435600, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 49.322781},
    {80.435600, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 51.458253},
    {80.435600, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 57.883240},
    {80.435600, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 65.921074},
    {80.435600, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 76.650626},
    {100.248000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 64.853552},
    {100.248000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 65.097010},
    {100.248000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 65.758732},
    {100.248000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 68.084161},
    {100.248000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 75.151581},
    {100.248000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 84.069285},
    {100.248000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 96.024020},
    {120.263000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 81.957940},
    {120.263000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 82.215043},
    {120.263000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 82.915088},
    {120.263000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 85.385085},
    {120.263000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 92.962528},
    {120.263000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 102.619584},
    {120.263000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 115.645671},
    {202.555000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 155.779813},
    {202.555000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 156.065895},
    {202.555000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 156.847402},
    {202.555000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 159.642731},
    {202.555000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 168.514010},
    {202.555000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 180.282417},
    {202.555000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 196.646414},
};

#define numberOptions ((int)(sizeof(test_data) / sizeof(struct test_data_type)))

static float tolerance = 0.003;

int main(int argc, char** argv) {
    // binomial tree fintech model...

    int retval = XLNX_OK;

    std::string path = std::string(argv[1]);
    m76 m76(path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    // Get a list of U200s available on the system (just because our current bitstreams are built for U200s)
    deviceList = DeviceManager::getDeviceList(device);

    if (deviceList.size() == 0) {
        printf("No matching devices found\n");
        exit(0);
    }

    printf("Found %zu matching devices\n", deviceList.size());

    // we'll just pick the first device in the...
    pChosenDevice = deviceList[0];

    if (retval == XLNX_OK) {
        // turn off trace output...turn it on here if you want extra debug output...
        Trace::setEnabled(true);
    }

    printf("\n\n\n");
    printf("[XF_FINTECH] HCF trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();

    retval = m76.claimDevice(pChosenDevice);

    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Multiple Options European Call [%d]\n", numberOptions);

        m76::m76_input_data inputData[numberOptions];
        float outputData[numberOptions];
        start = std::chrono::high_resolution_clock::now();

        // populate some data
        for (int i = 0; i < numberOptions; i++) {
            inputData[i].S = test_data[i].S;
            inputData[i].sigma = test_data[i].sigma;
            inputData[i].K = test_data[i].K;
            inputData[i].T = test_data[i].T;
            inputData[i].r = test_data[i].r;
            inputData[i].kappa = test_data[i].kappa;
            inputData[i].delta = test_data[i].delta;
            inputData[i].lambda = test_data[i].lambda;
        }

        retval = m76.run(inputData, outputData, numberOptions);

        end = std::chrono::high_resolution_clock::now();

        if (retval == XLNX_OK) {
            for (int i = 0; i < numberOptions; i++) {
                printf("[%02u] S=%f, K=%f, sigma=%f, r=%f, T=%f, kappa=%f, lambda=%f, delta=%f\n", i, inputData[i].S,
                       inputData[i].K, inputData[i].sigma, inputData[i].r, inputData[i].T, inputData[i].kappa,
                       inputData[i].lambda, inputData[i].delta);
                float diff = fabs(test_data[i].exp - outputData[i]);
                printf("    exp=%f, OptionPrice=%f, diff = %f\n", test_data[i].exp, outputData[i], diff);
                if (diff > tolerance) {
                    printf("    FAIL\n");
                    ret = 1;
                }
            }
            long long int executionTime =
                (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("[XF_FINTECH] ExecutionTime = %lld microseconds (average %lld microseconds)\n", executionTime,
                   executionTime / numberOptions);
        }
    }

    printf("[XF_FINTECH] M76 releasing device...\n");
    retval = m76.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
