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

int main(int argc, char** argv) {
    // hullwhite fintech model...
    std::string path = std::string(argv[1]);
    HullWhiteAnalytic hwa(path);

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
    printf("[XF_FINTECH] HullWhite trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();

    retval = hwa.claimDevice(pChosenDevice);

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
        static const int N_k0 = 16;
        static const int LEN = 16;

        double a = 0.10;
        double sigma = 0.01;

        double rates[LEN] = {0.0020, 0.0050, 0.0070, 0.0110, 0.0150, 0.0180,
                             0.0200, 0.0220, 0.0250, 0.0288, 0.0310, 0.0340};
        double times[LEN] = {0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 10.0, 20.0, 30.0};

        double t0[N_k0];
        double T0[N_k0];
        for (int i = 0; i < N_k0; i++) {
            t0[i] = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 15.0));
            T0[i] = t0[i] + 1.0 + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 15.0));
        }

        double outputP[LEN];

        printf("[XF_FINTECH] Multiple HullWhite Spread Calculations [%d]\n", N_k0);
        retval = hwa.run(a, sigma, times, rates, t0, T0, outputP);
        end = std::chrono::high_resolution_clock::now();

        if (retval == XLNX_OK) {
            for (int i = 0; i < N_k0; i++) {
                printf("[XF_FINTECH] [%02u] HullWhite Spread = %f\n", i, outputP[i]);

                // quick fix for pass fail criteria
                if (std::abs(outputP[N_k0 - 1] - 0.631695) > tolerance) {
                    printf("FAIL\n");
                    ret = 1;
                }
            }
            long long int executionTime =
                (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf(
                "[XF_FINTECH] ExecutionTime = %lld microseconds (average %lld "
                "microseconds)\n",
                executionTime, executionTime / N_k0);
        }
    }

    printf("[XF_FINTECH] HullWhite releasing device...\n");
    retval = hwa.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    }
    return ret;
}
