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
#include "test_vectors_bsm.hpp"

using namespace xf::fintech;

int check_result(float calculated, float expected, float tolerance) {
    if (std::abs(calculated - expected) > tolerance) {
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int retval = XLNX_OK;
    float tolerance = 0.01;

    std::string path = std::string(argv[1]);
    unsigned int numAssets = sizeof(test_data) / sizeof(struct test_data_type);
    CFBlackScholesMerton cfBlackScholesMerton(numAssets, path);

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

    retval = cfBlackScholesMerton.claimDevice(pChosenDevice);

    int ret = 0; // assume pass
    int numFails = 0;
    if (retval == XLNX_OK) {
        // Populate the asset data...
        for (unsigned int i = 0; i < numAssets; i++) {
            cfBlackScholesMerton.stockPrice[i] = test_data[i].s;
            cfBlackScholesMerton.strikePrice[i] = test_data[i].k;
            cfBlackScholesMerton.volatility[i] = test_data[i].v;
            cfBlackScholesMerton.riskFreeRate[i] = test_data[i].r;
            cfBlackScholesMerton.timeToMaturity[i] = test_data[i].t;
            cfBlackScholesMerton.dividendYield[i] = test_data[i].q;
        }

        ///////////////////
        // Run the model...
        ///////////////////
        cfBlackScholesMerton.run(OptionType::Call, numAssets);

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
            printf("[XLNX] | %5u | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f |\n", i,
                   cfBlackScholesMerton.optionPrice[i], cfBlackScholesMerton.delta[i], cfBlackScholesMerton.gamma[i],
                   cfBlackScholesMerton.vega[i], cfBlackScholesMerton.theta[i], cfBlackScholesMerton.rho[i]);

            if (!check_result(cfBlackScholesMerton.optionPrice[i], test_data[i].exp, tolerance)) {
                printf("[XLNX] expected(%8.5f), got (%8.5f)\n", test_data[i].exp, cfBlackScholesMerton.optionPrice[i]);
                numFails++;
                ret = 1;
            }
        }

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf("[XLNX] Processed %u assets in %lld us\n", numAssets, cfBlackScholesMerton.getLastRunTime());
    }

    cfBlackScholesMerton.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL: %d failures\n", numFails);
    }
    return ret;
}
