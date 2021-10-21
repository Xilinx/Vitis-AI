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

// The MC European XCLBIN contains 4 kernels.  Each of these kernels implement
// an instance of the MC European engine.
//
// In this example, we run all 4 kernels simultaneously, using the data for a
// SINGLE ASSET.
// i.e the same asset data is passed to all 4 kernels.
//
//
// Each of the kernels produce an independent output.  These outputs are then
// averaged to produce a final option price.

static double tolerance = 0.05;

static OptionType optionType = Put;

static const double initialStockPrice = 36.0;
static const double initialStrikePrice = 40.0;
static const double initialRiskFreeRate = 0.06;
static const double initialDividendYield = 0.0;
static const double initialVolatility = 0.20;
static const double initialTimeToMaturity = 1.0; /* in years */
static const double initialRequiredTolerance = 0.02;

/* The following variable is used to vary our input data for each run....*/
static const double varianceFactor = 0.001;

static double stockPrice;
static double strikePrice;
static double riskFreeRate;
static double dividendYield;
static double volatility;
static double timeToMaturity;
static double requiredTolerance;

/* The following variable will hold our calculated option price... */
static double optionPrice;

static void PrintParameters(void) {
    printf("\n");
    printf("\n");
    printf("[XLNX] ==========\n");
    printf("[XLNX] Parameters\n");
    printf("[XLNX] ==========\n");
    printf("[XLNX] initialOptionType        = %s\n", Trace::optionTypeToString(optionType));
    printf("[XLNX] initialStockPrice        = %f\n", initialStockPrice);
    printf("[XLNX] initialStrikePrice       = %f\n", initialStrikePrice);
    printf("[XLNX] initialRiskFreeRate      = %f\n", initialRiskFreeRate);
    printf("[XLNX] initialDividendYield     = %f\n", initialDividendYield);
    printf("[XLNX] initialVolatility        = %f\n", initialVolatility);
    printf("[XLNX] initialTimeToMaturity    = %f\n", initialTimeToMaturity);
    printf("[XLNX] initialRequiredTolerance = %f\n", initialRequiredTolerance);
    printf("\n");
    printf("[XLNX] varianceFactor           = %f * iteration\n", varianceFactor);
    printf("\n");
}

int MCDemoRunEuropeanSingle(Device* pChosenDevice, MCEuropean* pMCEuropean) {
    int retval = XLNX_OK;
    int i;
    int ret = 1; // assume fail
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    int NUM_ITERATIONS = 100;
    std::string mode_emu = "hw_emu";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    if (mode_emu == "hw_emu") {
        NUM_ITERATIONS = 1;
    }

    printf("\n\n\n");

    printf(
        "[XLNX] "
        "***************************************************************\n");
    printf("[XLNX] Running MC EUROPEAN SINGLE ASSET...\n");
    printf(
        "[XLNX] "
        "***************************************************************\n");

    //
    // Claim the device for our MCEuropean object...this will download the
    // required XCLBIN file (if needed)...
    //
    printf("[XLNX] mcEuropean trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();

    retval = pMCEuropean->claimDevice(pChosenDevice);

    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XLNX] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XLNX] ERROR- Failed to claim device - error = %d\n", retval);
    }

    if (retval == XLNX_OK) {
        PrintParameters();
    }

    //
    // Run the model a few times...
    //
    if (retval == XLNX_OK) {
        printf(
            "[XLNX] "
            "+-----------+-------------+--------------+-----------+------------+---"
            "---------+--------------+----------------+\n");
        printf(
            "[XLNX] | Iteration | Stock Price | Strike Price | Risk Free | Div. "
            "Yield | Volatility | Option Price | Execution Time |\n");
        printf(
            "[XLNX] "
            "+-----------+-------------+--------------+-----------+------------+---"
            "---------+--------------+----------------+\n");

        for (i = 0; i < NUM_ITERATIONS; i++) {
            /* We will apply some variance to our input data here so we are not
             * cacheing any values... */
            double variance = (1.0 + (varianceFactor * i));

            stockPrice = initialStockPrice * variance;
            strikePrice = initialStrikePrice * variance;
            riskFreeRate = initialRiskFreeRate * variance;
            dividendYield = initialDividendYield * variance;
            volatility = initialVolatility * variance;

            timeToMaturity = initialTimeToMaturity;
            requiredTolerance = initialRequiredTolerance;

            retval = pMCEuropean->run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility,
                                      timeToMaturity, requiredTolerance, &optionPrice);

            if (retval == XLNX_OK) {
                printf(
                    "[XLNX] | %9d | %11.4f | %12.4f | %9.4f | %10.4f | %10.4f | %12.4f "
                    "| %11lld us |\n",
                    i, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility, optionPrice,
                    pMCEuropean->getLastRunTime());
            } else {
                break; // out of loop
            }

            // quick fix to get pass/fail criteria
            if (i == 0 && std::abs(optionPrice - 3.8761) <= tolerance) {
                ret = 0;
            }
        }

        printf(
            "[XLNX] "
            "+-----------+-------------+--------------+-----------+------------+---"
            "---------+--------------+----------------+\n");
    }

    //
    // Release the device so another object can claim it...
    //
    printf("[XLNX] mcEuropean releasing device...\n");
    retval = pMCEuropean->releaseDevice();

    return ret;
}
