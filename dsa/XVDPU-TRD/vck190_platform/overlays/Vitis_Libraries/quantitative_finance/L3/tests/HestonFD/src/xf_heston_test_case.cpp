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

#include <tgmath.h>
#include <iostream>
#include <string>
#include <vector>

#include "xf_fintech_api.hpp"
#include "xf_fintech_heston_test_case.hpp"
#include "xf_fintech_results_csv.hpp"

using namespace std;
using namespace xf::fintech;

bool HestonFDTestCase::CompareValues(double val1, double val2) {
    // test based on value passed in via command line
    return std::abs(val1 - val2) < m_delta;
}

int HestonFDTestCase::Run(string testcase, string testscheme, std::vector<double> csvTableEntry) {
    // solver parameters
    double kappa;
    double eta;
    double sigma;
    double rho;
    double rd;
    double T;
    double K;
    double S;
    double V;

    // model parameters
    int N;
    int m1;
    int m2;

    double* priceGrid;
    double valueFromPriceGrid;
    bool testPass = true;
    int numberOfColumns;
    int numberOfRows;
    std::chrono::milliseconds duration;
    string goldenReferenceFile;
    int numberPriceGridMismatches = 0;

    CSVResults csvResults;
    double valueFromCSVResults;
    int row, col;
    ifstream infile;
    double maxDiff = 0.0;
    int maxIndex = 0;

    int retval;
    std::vector<double> pGrid;
    std::vector<double> sGrid;
    std::vector<double> vGrid;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    kappa = csvTableEntry[0];
    eta = csvTableEntry[1];
    sigma = csvTableEntry[2];
    rho = csvTableEntry[3];
    rd = csvTableEntry[4];
    // rf			= csvTableEntry[5];	//JIRA: DCA-133 - rf is no supported by
    // API.
    T = csvTableEntry[6];
    K = csvTableEntry[7];
    S = csvTableEntry[8];
    V = csvTableEntry[9];
    // theta		= csvTableEntry[10];	/JIRA: DCA-133 - theta is no supported by
    // API.
    N = (int)csvTableEntry[11];
    m1 = (int)csvTableEntry[12] + 1;
    m2 = (int)csvTableEntry[13] + 1;

    std::cout << std::setprecision(4) << std::endl;
    std::cout << "Input Paramaters:" << std::endl;
    std::cout << "testcase " << testcase << std::endl;
    std::cout << "testscheme " << testscheme << std::endl;
    std::cout << "kappa " << kappa << std::endl;
    std::cout << "eta " << eta << std::endl;
    std::cout << "sigma " << sigma << std::endl;
    std::cout << "rho " << rho << std::endl;
    std::cout << "rd " << rd << std::endl;
    std::cout << "T " << T << std::endl;
    std::cout << "K " << K << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "V " << V << std::endl;
    std::cout << "N " << N << std::endl;
    std::cout << "m1 " << m1 << std::endl;
    std::cout << "m2 " << m2 << std::endl;

    std::cout << std::endl;

    FDHeston fdHeston(m1, m2, m_xclbin);

    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    deviceList = DeviceManager::getDeviceList(m_device);

    if (deviceList.size() == 0) {
        printf("No matching devices found\n");
        exit(0);
    }

    std::cout << "Found " << deviceList.size() << " matching devices" << std::endl;

    // we'll just pick the first device in the list...
    pChosenDevice = deviceList[0];

    fdHeston.claimDevice(pChosenDevice);

    // Read in golden result file
    goldenReferenceFile = testcase + "_" + std::to_string(m1) + "x" + std::to_string(m2) + ".csv";
    std::cout << "Reading Golden Reference file " << goldenReferenceFile << std::endl;
    infile.open("data/" + goldenReferenceFile, std::ifstream::in);

    if (infile.is_open()) {
        // PAUSE();

        start = std::chrono::high_resolution_clock::now();

        retval = fdHeston.run(S, K, rd, V, T, kappa, sigma, rho, eta, N, pGrid, sGrid, vGrid);

        end = std::chrono::high_resolution_clock::now();

        std::cout << "Duration - " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms. Result - " << retval << std::endl;

        csvResults.init(infile);

        // Compare the results with the golden result file
        priceGrid = pGrid.data();
        numberOfRows = csvResults.showNumberOfRows();
        numberOfColumns = csvResults.showNumberOfColumns();

        for (row = 0; row < m2; row++) {
            // ignore row if the golden results grid is smaller than the solver
            // results
            if (row < numberOfRows) {
                for (col = 0; col < m1; col++) {
                    // get next value
                    valueFromPriceGrid = *(priceGrid++);

                    // ignore column entry if the golden results grid is smaller than the
                    // solver results
                    if (col < numberOfColumns) {
                        // check for NaN value
                        if (std::isnan(valueFromPriceGrid) == true) {
                            std::cout << "Nan value found row:" << row << " col:" << col
                                      << " Kernel:" << valueFromPriceGrid << std::endl;
                            numberPriceGridMismatches++;
                        } else {
                            valueFromCSVResults = csvResults.data(row)[col];

                            double diff = valueFromPriceGrid - valueFromCSVResults;
                            if (std::abs(diff) > std::abs(maxDiff)) {
                                maxDiff = diff;
                                maxIndex = (m1 * row) + col + 1;
                            }

                            if (CompareValues(valueFromCSVResults, valueFromPriceGrid) != true) {
                                std::cout << std::setprecision(10) << "Result differs for row:" << row << " col:" << col
                                          << " Reference:" << valueFromCSVResults << " Kernel:" << valueFromPriceGrid
                                          << std::endl;
                                numberPriceGridMismatches++;
                            }
                        }
                    }
                }
            }
        }

        std::cout << std::setprecision(5) << "Maximum difference is " << maxDiff << ", found at array index "
                  << maxIndex << std::endl;

        if (numberPriceGridMismatches > 0) {
            testPass = false;
        }

        if (testPass) {
            std::cout << testcase << " PASSED" << std::endl;
        } else {
            std::cout << testcase << " FAILED" << std::endl;
            std::cout << "################################################################" << std::endl << std::endl;
        }
    } else {
        std::cout << "Failed to open " << goldenReferenceFile << std::endl;
        std::cout << testcase << " FAILED" << std::endl;
        throw;
    }

    fdHeston.releaseDevice();

    return numberPriceGridMismatches;
}
