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
#include "test_vectors_quanto.hpp"

using namespace xf::fintech;

static float tolerance = 0.001;
int check(float calculated, float expected, float tolerance) {
    if (std::abs(calculated - expected) > tolerance) {
        printf("ERROR: expected %0.6f, got %0.6f\n", expected, calculated);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int retval = XLNX_OK;
    float tolerance = 0.001;

    std::string path = std::string(argv[1]);

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

    unsigned int numAssets = sizeof(test_data) / sizeof(struct test_data_type);
    CFQuanto cfQuanto(numAssets, path);

    retval = cfQuanto.claimDevice(pChosenDevice);

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        // Populate the asset data...
        for (unsigned int i = 0; i < numAssets; i++) {
            cfQuanto.stockPrice[i] = test_data[i].s;
            cfQuanto.strikePrice[i] = test_data[i].k;
            cfQuanto.volatility[i] = test_data[i].v;
            cfQuanto.timeToMaturity[i] = test_data[i].t;
            cfQuanto.domesticRate[i] = test_data[i].rd;
            cfQuanto.foreignRate[i] = test_data[i].rf;
            cfQuanto.dividendYield[i] = test_data[i].q;
            cfQuanto.exchangeRate[i] = test_data[i].E;
            cfQuanto.exchangeRateVolatility[i] = test_data[i].fxv;
            cfQuanto.correlation[i] = test_data[i].corr;
        }

        ///////////////////
        // Run the model...
        ///////////////////
        cfQuanto.run(OptionType::Call, numAssets);

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
            printf("[XLNX] | %5u | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f |\n", i, cfQuanto.optionPrice[i],
                   cfQuanto.delta[i], cfQuanto.gamma[i], cfQuanto.vega[i], cfQuanto.theta[i], cfQuanto.rho[i]);
        }

        // quick fix to get pass/fail criteria
        if (!check(cfQuanto.optionPrice[numAssets - 2], 47.74052, tolerance)) {
            ret = 1;
        }
        if (!check(cfQuanto.delta[numAssets - 2], 0.68087, tolerance)) {
            ret = 1;
        }
        if (!check(cfQuanto.gamma[numAssets - 2], 0.00238, tolerance)) {
            ret = 1;
        }
        if (!check(cfQuanto.vega[numAssets - 2], 0.34321, tolerance)) {
            ret = 1;
        }
        if (!check(cfQuanto.theta[numAssets - 2], -0.03394, tolerance)) {
            ret = 1;
        }
        if (!check(cfQuanto.rho[numAssets - 2], 0.33964, tolerance)) {
            ret = 1;
        }

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf("[XLNX] Processed %u assets in %lld us\n", numAssets, cfQuanto.getLastRunTime());
        printf("[XLNX] Comparing calculated value against expected value at a tolerance of %8.5f\n", tolerance);
        printf("[XLNX] Total tests = %d\n", numAssets);
    }

    cfQuanto.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
