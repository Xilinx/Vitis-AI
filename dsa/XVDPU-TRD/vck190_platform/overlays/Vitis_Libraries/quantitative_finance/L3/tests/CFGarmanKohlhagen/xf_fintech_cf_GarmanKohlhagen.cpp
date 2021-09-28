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
#include "test_vectors_gk.hpp"

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
    int retval = XLNX_OK;

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
    std::cout << "sizeof(test_data) = " << sizeof(test_data) << std::endl;
    std::cout << "sizeof(struct test_data_type) = " << sizeof(struct test_data_type) << std::endl;
    std::cout << "sizeof(test_data) / sizeof(struct test_data_type) = "
              << sizeof(test_data) / sizeof(struct test_data_type) << std::endl;
    std::cout << "numAssets = " << numAssets << std::endl;
    CFGarmanKohlhagen cfGarmanKohlhagen(numAssets, path);

    retval = cfGarmanKohlhagen.claimDevice(pChosenDevice);

    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        // Populate the asset data...
        for (unsigned int i = 0; i < numAssets; i++) {
            cfGarmanKohlhagen.stockPrice[i] = test_data[i].s;
            cfGarmanKohlhagen.strikePrice[i] = test_data[i].k;
            cfGarmanKohlhagen.volatility[i] = test_data[i].v;
            cfGarmanKohlhagen.timeToMaturity[i] = test_data[i].t;
            cfGarmanKohlhagen.domesticRate[i] = test_data[i].r_domestic;
            cfGarmanKohlhagen.foreignRate[i] = test_data[i].r_foreign;
        }

        ///////////////////
        // Run the model...
        ///////////////////
        cfGarmanKohlhagen.run(OptionType::Call, numAssets);

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
                   cfGarmanKohlhagen.optionPrice[i], cfGarmanKohlhagen.delta[i], cfGarmanKohlhagen.gamma[i],
                   cfGarmanKohlhagen.vega[i], cfGarmanKohlhagen.theta[i], cfGarmanKohlhagen.rho[i]);
        }

        // quick fix to get pass/fail criteria
        if (!check(cfGarmanKohlhagen.optionPrice[numAssets - 1], 16.63863, tolerance)) {
            ret = 1;
        }
        if (!check(cfGarmanKohlhagen.delta[numAssets - 1], 0.59054, tolerance)) {
            ret = 1;
        }
        if (!check(cfGarmanKohlhagen.gamma[numAssets - 1], 0.00869, tolerance)) {
            ret = 1;
        }
        if (!check(cfGarmanKohlhagen.vega[numAssets - 1], 0.16683, tolerance)) {
            ret = 1;
        }
        if (!check(cfGarmanKohlhagen.theta[numAssets - 1], -0.07054, tolerance)) {
            ret = 1;
        }
        if (!check(cfGarmanKohlhagen.rho[numAssets - 1], 0.09181, tolerance)) {
            ret = 1;
        }

        printf(
            "[XLNX] "
            "+-------+----------+----------+----------+----------+----------+------"
            "----+\n");
        printf("[XLNX] Processed %u assets in %lld us\n", numAssets, cfGarmanKohlhagen.getLastRunTime());
    }

    cfGarmanKohlhagen.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
