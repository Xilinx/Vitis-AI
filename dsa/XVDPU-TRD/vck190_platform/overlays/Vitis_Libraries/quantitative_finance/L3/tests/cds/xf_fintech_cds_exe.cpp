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

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <vector>

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

static float tolerance = 0.001;

static int check(float actual, float expected, float tol) {
    int ret = 1; // assume pass
    if (std::abs(actual - expected) > tol) {
        printf("ERROR: expected %0.6f, got %0.6f\n", expected, actual);
        ret = 0;
    }
    return ret;
}

int main(int argc, char** argv) {
    // credit default swap fintech model...
    std::string path = std::string(argv[1]);
    CreditDefaultSwap cds(path);

    int retval = XLNX_OK;

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    // device passed in via compile
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
    printf("[XF_FINTECH] CreditDefaultSwap trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();

    retval = cds.claimDevice(pChosenDevice);

    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        // note these values should match the generated kernel
        static const int IRLEN = 21;
        static const int HAZARDLEN = 6;
        static const int N = 8;

        float ratesIR[IRLEN] = {0.0300, 0.0335, 0.0366, 0.0394, 0.0418, 0.0439, 0.0458, 0.0475, 0.0490, 0.0503, 0.0514,
                                0.0524, 0.0533, 0.0541, 0.0548, 0.0554, 0.0559, 0.0564, 0.0568, 0.0572, 0.0575};

        float timesIR[IRLEN] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                                5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};

        float ratesHazard[HAZARDLEN] = {0.005, 0.01, 0.01, 0.015, 0.010, 0.010};
        float timesHazard[HAZARDLEN] = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
        float maturity[N] = {2.0, 3.0, 4.0, 5.55, 6.33, 7.27, 8.001, 9.999};
        int frequency[N] = {4, 12, 2, 1, 12, 4, 1, 12};
        float recovery[N] = {0.15, 0.67, 0.22, 0.01, 0.80, 0.99, 0.001, 0.44};
        float nominal[N] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        float CDSSpread[N];

        // quick fix to get pass/fail criteria
        float expectedCDS[N] = {0.010623, 0.003858, 0.008916, 0.010177, 0.002179, 0.000104, 0.009482, 0.005963};

        printf("[XF_FINTECH] Multiple CDS Spread Calculations [%d]\n", N);
        retval = cds.run(timesIR, ratesIR, timesHazard, ratesHazard, nominal, recovery, maturity, frequency, CDSSpread);
        end = std::chrono::high_resolution_clock::now();

        if (retval == XLNX_OK) {
            for (int i = 0; i < N; i++) {
                printf("[XF_FINTECH] [%02u] CDS Spread = %f\n", i, CDSSpread[i]);
                if (!check(CDSSpread[i], expectedCDS[i], tolerance)) {
                    ret = 1;
                }
            }
            long long int executionTime =
                (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf(
                "[XF_FINTECH] ExecutionTime = %lld microseconds (average %lld "
                "microseconds)\n",
                executionTime, executionTime / N);
        }
    }

    printf("[XF_FINTECH] Credit Default Swap releasing device...\n");
    retval = cds.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
