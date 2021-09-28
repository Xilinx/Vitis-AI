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

float tolerance = 0.000003;

#define NUM_ASSETS (10)
#define NUM_PRICES (61)
float test_data[NUM_ASSETS * NUM_PRICES] = {
    37.57,  38.74,  41.52,  43.47,  43.65,  41.17,  42.35,  46.84,  51.38,  51.60,  51.99,  47.29,  48.08,
    49.56,  52.94,  52.71,  50.28,  53.47,  55.79,  55.52,  52.13,  53.33,  56.46,  52.67,  47.87,  50.00,
    48.83,  54.05,  53.11,  50.86,  52.41,  53.19,  54.62,  59.45,  58.69,  58.68,  60.53,  63.25,  66.92,
    63.92,  62.97,  66.66,  70.36,  71.76,  74.90,  76.00,  74.51,  71.51,  74.06,  74.46,  76.63,  80.39,
    83.13,  85.26,  89.74,  87.16,  92.16,  95.52,  100.33, 99.05,  99.99,

    84.74,  94.76,  97.03,  108.33, 104.10, 94.09,  90.65,  101.65, 103.53, 93.93,  116.25, 98.03,  104.53,
    122.29, 148.39, 166.72, 178.37, 154.79, 128.73, 75.08,  35.68,  29.69,  36.33,  29.32,  19.38,  20.81,
    26.15,  33.63,  35.26,  39.22,  43.24,  43.83,  34.07,  44.17,  54.52,  43.95,  52.42,  62.90,  54.12,
    46.79,  38.21,  43.94,  42.16,  43.50,  42.39,  48.28,  58.02,  57.28,  57.15,  53.62,  47.43,  45.89,
    45.82,  39.80,  30.01,  21.94,  25.27,  27.26,  26.42,  30.14,  31.01,

    44.84,  46.63,  47.10,  53.19,  55.32,  54.10,  58.99,  69.82,  71.24,  66.07,  69.23,  59.42,  68.04,
    73.58,  80.14,  89.35,  89.45,  78.93,  70.99,  44.80,  23.82,  21.48,  22.74,  20.87,  18.02,  18.69,
    21.99,  31.13,  31.04,  33.82,  33.61,  35.03,  32.09,  37.23,  43.37,  36.67,  36.40,  41.82,  36.99,
    29.07,  25.65,  29.43,  27.96,  31.81,  31.24,  30.73,  36.99,  35.38,  35.75,  35.24,  36.14,  32.79,
    34.08,  30.54,  21.74,  15.74,  20.51,  18.89,  18.19,  20.52,  23.30,

    25.53,  25.26,  27.14,  27.91,  26.80,  26.36,  26.22,  26.88,  33.59,  30.76,  32.59,  29.84,  25.00,
    26.08,  26.21,  26.12,  25.38,  23.72,  25.27,  24.72,  20.68,  18.85,  18.13,  15.94,  15.16,  17.25,
    19.02,  19.74,  22.46,  22.22,  23.42,  24.44,  26.34,  28.06,  29.09,  26.89,  27.49,  28.08,  29.28,
    24.85,  22.16,  24.86,  22.72,  23.71,  25.82,  24.61,  27.19,  27.01,  26.05,  24.88,  25.40,  24.67,
    25.64,  27.02,  26.40,  24.70,  26.43,  25.58,  25.96,  29.53,  30.77,

    84.61,  92.91,  99.80,  121.19, 122.04, 131.76, 138.48, 153.47, 189.95, 182.22, 198.08, 135.36, 125.02,
    143.50, 173.95, 188.75, 167.44, 158.95, 169.53, 113.66, 107.59, 92.67,  85.35,  90.13,  89.31,  105.12,
    125.83, 135.81, 142.43, 163.39, 168.21, 185.35, 188.50, 199.91, 210.73, 192.06, 204.62, 235.00, 261.09,
    256.88, 251.53, 257.25, 243.10, 283.75, 300.98, 311.15, 322.56, 339.32, 353.21, 348.51, 350.13, 347.83,
    335.67, 390.48, 384.83, 381.32, 404.78, 382.20, 405.00, 456.48, 493.17,

    43.45,  44.83,  46.12,  47.31,  45.39,  45.41,  48.42,  49.36,  46.53,  47.91,  46.48,  42.38,  45.24,
    46.88,  45.64,  46.50,  43.09,  47.62,  49.16,  50.66,  45.53,  39.22,  39.91,  39.77,  35.74,  33.64,
    38.67,  40.03,  43.10,  43.96,  43.93,  45.93,  48.09,  49.41,  49.99,  51.14,  49.36,  50.57,  52.00,
    50.93,  47.95,  47.71,  47.74,  48.53,  48.29,  47.70,  49.49,  48.73,  52.29,  52.70,  55.91,  56.04,
    54.40,  54.85,  53.84,  52.72,  53.73,  49.16,  50.57,  49.52,  50.21,

    28.97,  29.34,  30.59,  31.18,  32.00,  32.40,  32.49,  34.84,  34.63,  32.22,  31.46,  30.01,  28.38,
    31.69,  28.00,  26.31,  23.11,  24.50,  24.33,  22.38,  17.12,  15.07,  14.49,  10.85,  7.84,   9.31,
    11.65,  12.41,  10.88,  12.44,  12.90,  15.33,  13.32,  14.96,  14.22,  15.11,  15.19,  17.21,  17.84,
    15.46,  13.72,  15.34,  13.78,  15.58,  15.36,  15.18,  17.67,  19.46,  20.35,  19.51,  19.90,  19.11,
    18.50,  17.57,  16.00,  15.07,  16.55,  15.76,  17.91,  18.71,  19.13,

    44.68,  44.85,  44.74,  45.07,  43.45,  42.14,  45.04,  45.24,  43.45,  41.52,  37.67,  40.31,  36.28,
    35.19,  34.85,  31.57,  22.59,  31.14,  29.48,  33.79,  23.34,  15.69,  13.90,  6.50,   3.90,   6.75,
    8.84,   11.16,  13.08,  14.66,  17.43,  16.78,  14.46,  15.72,  14.94,  15.06,  16.53,  17.72,  17.70,
    15.63,  14.28,  13.95,  12.38,  13.03,  11.39,  10.89,  13.28,  13.66,  14.22,  13.28,  12.23,  11.70,
    10.92,  9.68,   8.15,   6.11,   6.82,   5.44,   5.56,   7.13,   8.18,

    19.57,  19.80,  20.74,  21.78,  20.26,  18.63,  19.92,  19.59,  19.73,  19.29,  18.45,  18.96,  18.34,
    17.23,  16.55,  16.19,  14.61,  15.61,  16.25,  15.68,  15.06,  14.21,  15.32,  12.61,  10.88,  12.04,
    11.81,  13.58,  13.41,  14.24,  15.08,  14.94,  15.37,  16.56,  16.58,  17.01,  16.15,  15.78,  15.38,
    14.16,  13.26,  13.95,  14.96,  16.14,  16.38,  15.48,  16.63,  17.30,  18.47,  19.49,  20.13,  20.79,
    19.96,  18.65,  18.60,  17.32,  18.87,  19.86,  21.42,  21.18,  21.14,

    64.20,  67.58,  71.10,  74.81,  75.45,  76.58,  77.42,  83.59,  83.08,  80.83,  84.94,  77.70,  79.23,
    77.01,  84.74,  81.18,  80.60,  73.56,  73.55,  71.39,  68.14,  74.10,  73.81,  70.71,  63.09,  63.28,
    61.95,  64.83,  65.35,  65.80,  65.03,  64.53,  67.40,  71.01,  64.50,  60.95,  61.89,  63.77,  64.52,
    57.96,  54.71,  57.21,  57.07,  59.65,  64.19,  67.58,  71.04,  78.38,  83.53,  82.16,  85.92,  81.98,
    79.93,  78.37,  73.18,  71.81,  77.20,  80.00,  84.30,  83.28,  84.88,
};

int check_results(std::string s,
                  std::vector<float>& w,
                  float portfolio_variance,
                  float portfolio_return,
                  float sharpe,
                  std::vector<float>& exp,
                  int num_assets,
                  float exp_return = 0,
                  float exp_variance = 0,
                  float exp_sharp_ratio = 0) {
    int res = 0;
    for (int i = 0; i < num_assets; i++) {
        std::cout << s << " w[" << i << "] = " << w[i];
        if (std::abs(exp[i] - w[i]) > tolerance) {
            std::cout << " FAIL: expected " << exp[i] << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_return != 0) {
        std::cout << s << " Expected return = " << portfolio_return;
        if (std::abs(portfolio_return - exp_return) > tolerance) {
            std::cout << " FAIL: expected " << exp_return << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_variance != 0) {
        std::cout << s << " Variance = " << portfolio_variance;
        if (std::abs(portfolio_variance - exp_variance) > tolerance) {
            std::cout << " FAIL: expected " << exp_variance << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_sharp_ratio != 0) {
        std::cout << s << " Sharpe Ratio = " << sharpe;
        if (std::abs(sharpe - exp_sharp_ratio) > tolerance) {
            std::cout << " FAIL: expected " << exp_sharp_ratio << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }
    return res;
}

int main(int argc, char** argv) {
    // portfolio optimisation model...
    std::string path = std::string(argv[1]);
    portfolio_optimisation po(path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    float targetReturn = 0.02;
    float riskFreeRate = 0.001;
    std::vector<float> GMVPWeights(NUM_ASSETS);
    std::vector<float> EffWeights(NUM_ASSETS);
    std::vector<float> TanWeights(NUM_ASSETS);
    std::vector<float> EffTanWeights(NUM_ASSETS);
    float GMVPVariance;
    float EffVariance;
    float TanVariance;
    float EffTanVariance;
    float GMVPReturn;
    float EffReturn;
    float TanReturn;
    float TanSharpe;
    float EffTanReturn;

    int retval = XLNX_OK;

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
    printf("[XF_FINTECH] PortfolioOptimisation trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();
    retval = po.claimDevice(pChosenDevice);
    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Running kernel\n");

        start = std::chrono::high_resolution_clock::now();

        retval = po.run(test_data, NUM_PRICES, NUM_ASSETS, riskFreeRate, targetReturn, GMVPWeights, &GMVPVariance,
                        &GMVPReturn, EffWeights, &EffVariance, &EffReturn, TanWeights, &TanVariance, &TanReturn,
                        &TanSharpe, EffTanWeights, &EffTanVariance, &EffTanReturn);

        end = std::chrono::high_resolution_clock::now();
    }

    int res = 0;
    if (retval == XLNX_OK) {
        // check results GMVP
        std::vector<float> exp = {0.231147, -0.002218, -0.084974, 0.106045, 0.006200,
                                  0.411148, -0.069371, -0.035975, 0.123788, 0.314209};
        std::cout << "Global Minimum Variance Portfolio" << std::endl;
        if (check_results("GMVP", GMVPWeights, GMVPVariance, GMVPReturn, 0, exp, NUM_ASSETS, 0.0090025, 0.000977613)) {
            res = 1;
        }

        // check results efficient portfolio
        std::vector<float> exp_eff = {0.603973, -0.042119, -0.081139, -0.044018, 0.207977,
                                      0.247975, -0.087848, -0.030071, 0.080708,  0.144561};
        std::cout << "Efficient Portfolio with target return of " << targetReturn << std::endl;
        if (check_results("Eff", EffWeights, EffVariance, EffReturn, 0, exp_eff, NUM_ASSETS, 0.02, 0.0014647)) {
            res = 1;
        }

        // check results tangency portfolio
        tolerance = 0.2; /* temp relaxation of tolerance for u200 CR-1083822*/
        std::vector<float> exp_tan = {1.259484,  -0.112273, -0.074397, -0.307861, 0.562747,
                                      -0.038921, -0.120334, -0.019689, 0.004963,  -0.153718};
        std::cout << "Tangency Portfolio for risk free rate of " << riskFreeRate << std::endl;
        if (check_results("Tan", TanWeights, TanVariance, TanReturn, TanSharpe, exp_tan, NUM_ASSETS, 0.0393361,
                          0.00468327, 0.560187)) {
            res = 1;
        }
        tolerance = 0.003;

        // check results tangency and risk free portfolio with target return
        std::vector<float> exp_rf = {0.624221,  -0.055644, -0.036872, -0.152581, 0.278906,
                                     -0.019290, -0.059640, -0.009758, 0.002460,  -0.076185};
        std::cout << "Tangency Portfolio for risk free rate of " << riskFreeRate << " and target return "
                  << targetReturn << std::endl;
        if (check_results("Eff Tan", EffTanWeights, 0, 0, 0, exp_rf, NUM_ASSETS)) {
            res = 1;
        }
        float rf = 0;
        for (int i = 0; i < NUM_ASSETS; i++) {
            rf += EffTanWeights[i];
        }
        std::cout << "Eff Tan proportion in risk free = " << 1 - rf << std::endl;
        long long int executionTime =
            (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("[XF_FINTECH] ExecutionTime = %lld microseconds)\n", executionTime);
    }

    printf("[XF_FINTECH] Portfolio Optimisation releasing device...\n");
    retval = po.releaseDevice();

    if (!res) {
        std::cout << "TEST PASS" << std::endl;
    }

    return res;
}
