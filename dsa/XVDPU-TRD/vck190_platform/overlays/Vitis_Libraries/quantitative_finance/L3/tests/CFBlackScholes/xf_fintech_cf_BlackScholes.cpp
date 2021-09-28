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

static const unsigned int numAssets = 100;

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
    CFBlackScholes cfBlackScholes(numAssets, path);

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

    retval = cfBlackScholes.claimDevice(pChosenDevice);

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        // Populate the asset data...
        for (unsigned int i = 0; i < numAssets; i++) {
            cfBlackScholes.stockPrice[i] = 100.0f;
            cfBlackScholes.strikePrice[i] = 100.0f;
            cfBlackScholes.volatility[i] = 0.1f;
            cfBlackScholes.riskFreeRate[i] = 0.025f;
            cfBlackScholes.timeToMaturity[i] = 1.0f;
        }

        ///////////////////
        // Run the model...
        ///////////////////
        cfBlackScholes.run(OptionType::Put, numAssets);

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
            printf("[XLNX] | %5u | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f |\n", i, cfBlackScholes.optionPrice[i],
                   cfBlackScholes.delta[i], cfBlackScholes.gamma[i], cfBlackScholes.vega[i], cfBlackScholes.theta[i],
                   cfBlackScholes.rho[i]);

            // quick fix to get pass/fail criteria
            if (!check(cfBlackScholes.optionPrice[i], 2.82636, tolerance)) {
                ret = 1;
            }
            if (!check(cfBlackScholes.delta[i], -0.38209, tolerance)) {
                ret = 1;
            }
            if (!check(cfBlackScholes.gamma[i], 0.03814, tolerance)) {
                ret = 1;
            }
            if (!check(cfBlackScholes.vega[i], 0.38139, tolerance)) {
                ret = 1;
            }
            if (!check(cfBlackScholes.theta[i], -0.00241, tolerance)) {
                ret = 1;
            }
            if (!check(cfBlackScholes.rho[i], 0.41035, tolerance)) {
                ret = 1;
            }
        }

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf("[XLNX] Processed %u assets in %lld us\n", numAssets, cfBlackScholes.getLastRunTime());
    }

    cfBlackScholes.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
