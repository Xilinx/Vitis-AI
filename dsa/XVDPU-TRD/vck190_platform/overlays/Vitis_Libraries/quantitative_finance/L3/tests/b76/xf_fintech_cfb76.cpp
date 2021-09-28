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

static const unsigned int numAssets = 1000;
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
    int retval = XLNX_OK;

    std::string path = std::string(argv[1]);
    CFB76 cfB76(numAssets, path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    // Get a list of U250s available on the system (just because our current
    // bitstreams are built for U250s)
    deviceList = DeviceManager::getDeviceList(device);

    if (deviceList.size() == 0) {
        printf("[XLNX] No matching devices found\n");
        exit(0);
    }

    printf("[XLNX] Found %zu matching devices\n", deviceList.size());

    // we'll just pick the first device in the...
    pChosenDevice = deviceList[0];

    retval = cfB76.claimDevice(pChosenDevice);

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        // Populate the asset data...
        for (unsigned int i = 0; i < numAssets; i++) {
            cfB76.forwardPrice[i] = 100.0f;
            cfB76.strikePrice[i] = 100.0f;
            cfB76.volatility[i] = 0.1f;
            cfB76.riskFreeRate[i] = 0.025f;
            cfB76.timeToMaturity[i] = 1.0f;
        }

        ///////////////////
        // Run the model...
        ///////////////////
        cfB76.run(OptionType::Put, numAssets);

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf(
            "[XLNX] | Index |  Price   |  Delta   |  Gamma   |   Vega   |  Theta   "
            "|   Rho    |\n");
        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");

        for (unsigned int i = 0; i < numAssets; i++) {
            printf("[XLNX] | %5u | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f |\n", i, cfB76.optionPrice[i],
                   cfB76.delta[i], cfB76.gamma[i], cfB76.vega[i], cfB76.theta[i], cfB76.rho[i]);

            // quick fix to get pass/fail criteria
            if (!check(cfB76.optionPrice[i], 3.88931, tolerance)) {
                ret = 1;
            }
            if (!check(cfB76.delta[i], -0.46821, tolerance)) {
                ret = 1;
            }
            if (!check(cfB76.gamma[i], 0.03886, tolerance)) {
                ret = 1;
            }
            if (!check(cfB76.vega[i], 0.38861, tolerance)) {
                ret = 1;
            }
            if (!check(cfB76.theta[i], -0.00506, tolerance)) {
                ret = 1;
            }
            if (!check(cfB76.rho[i], 0.50710, tolerance)) {
                ret = 1;
            }
        }

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf("[XLNX] Processed %u assets in %lld us\n", numAssets, cfB76.getLastRunTime());
    }

    cfB76.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    return ret;
}
