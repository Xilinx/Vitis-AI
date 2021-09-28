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

std::chrono::time_point<std::chrono::high_resolution_clock> start;
std::chrono::time_point<std::chrono::high_resolution_clock> end;
long long int duration;

static const double DOW_DIVISOR = 0.14748071991788;

double STOCK_PRICES[] = {
    163.69, // MMM
    117.38, // AXP
    182.06, // AAPL
    347.82, // BA
    122.38, // CAT
    116.65, // CVX
    54.00,  // CSCO
    50.64,  // KO
    135.32, // DIS
    49.53,  // DOW
    72.77,  // XOM
    187.86, // GS
    194.16, // HD
    131.49, // IBM
    44.27,  // INTC
    134.14, // JNJ
    109.09, // JPM
    199.87, // MCD
    81.94,  // MRK
    124.86, // MSFT
    82.31,  // NKE
    42.67,  // PFE
    106.31, // PG
    148.56, // TRV
    130.23, // UTX
    244.16, // UNH
    57.34,  // VZ
    163.21, // V
    103.97, // WMT
    50.81   // WBA
};

const unsigned int NUM_ASSETS = sizeof(STOCK_PRICES) / sizeof(STOCK_PRICES[0]);

OptionType optionTypes[NUM_ASSETS];
double strikePrices[NUM_ASSETS];
double riskFreeRates[NUM_ASSETS];
double volatility[NUM_ASSETS];
double dividendYields[NUM_ASSETS];
double timeToMaturity[NUM_ASSETS];
double tolerance = 0.05;
double requiredTolerance[NUM_ASSETS];

double DJIA;

void InitialiseAssetArrays(void) {
    unsigned int i;

    for (i = 0; i < NUM_ASSETS; i++) {
        optionTypes[i] = OptionType::Put;
        strikePrices[i] = 0.0;
        riskFreeRates[i] = 0.03;
        volatility[i] = 0.20;
        dividendYields[i] = 0.0;
        timeToMaturity[i] = 1.0;
        requiredTolerance[i] = tolerance;
    }
}

int main(int argc, char** argv) {
    int retval = XLNX_OK;

    std::string path = std::string(argv[1]);
    MCEuropeanDJE mcEuropeanDJE(path);

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

    if (retval == XLNX_OK) {
        // turn off trace output...turn it on here if you want extra debug output...
        Trace::setEnabled(false);
    }

    if (retval == XLNX_OK) {
        InitialiseAssetArrays();
    }

    if (retval == XLNX_OK) {
        //
        // Claim the device for our MCEuropeanDJE object...this will download the
        // required XCLBIN file (if needed)...
        //
        printf("[XLNX] mcEuropeanDJE trying to claim device...\n");

        start = std::chrono::high_resolution_clock::now();

        retval = mcEuropeanDJE.claimDevice(pChosenDevice);

        end = std::chrono::high_resolution_clock::now();

        if (retval == XLNX_OK) {
            duration = (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("[XLNX] Device setup time = %lld microseconds\n", duration);
        } else {
            printf("[XLNX] ERROR- Failed to claim device - error = %d\n", retval);
        }
    }

    double expected_value = 261.9595;
    if (retval == XLNX_OK) {
        printf("[XLNX] Running MCEuropeanDJE...\n");

        std::string mode_emu = "hw";
        if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
            mode_emu = std::getenv("XCL_EMULATION_MODE");
        }
        std::cout << "[XLNX] Running in " << mode_emu << " mode" << std::endl;

        if (mode_emu.compare("hw_emu") == 0 || mode_emu.compare("sw_emu") == 0) {
            // emulation limit to single asset and limited number of samples
            unsigned int numAssets = 1;
            double requiredSamples = 256;
            expected_value = 11.4370;

            retval = mcEuropeanDJE.run(optionTypes, STOCK_PRICES, strikePrices, riskFreeRates, dividendYields,
                                       volatility, timeToMaturity, &requiredSamples, numAssets, DOW_DIVISOR, &DJIA);
        } else {
            retval = mcEuropeanDJE.run(optionTypes, STOCK_PRICES, strikePrices, riskFreeRates, dividendYields,
                                       volatility, timeToMaturity, requiredTolerance, NUM_ASSETS, DOW_DIVISOR, &DJIA);
        }
    }

    if (retval == XLNX_OK) {
        printf("[XLNX] DJIA = %8.4f, Execution Time = %lld us\n", DJIA, mcEuropeanDJE.getLastRunTime());
    }

    printf("[XLNX] mcEuropeanDJE releasing device...\n");
    mcEuropeanDJE.releaseDevice();

    // quick fix to get pass/fail criteria
    int ret = 0; // assume pass
    if (std::abs(DJIA - expected_value) > tolerance) {
        printf("FAIL\n");
        ret = 1;
    } else {
        printf("PASS\n");
    }

    return ret;
}
