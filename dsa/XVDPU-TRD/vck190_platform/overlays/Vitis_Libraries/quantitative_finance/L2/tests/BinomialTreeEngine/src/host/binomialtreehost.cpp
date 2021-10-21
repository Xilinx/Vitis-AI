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

/**
 * @file binomialtreehost.cpp
 * @brief Host code used to verify engine.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>
#include <algorithm>

#include "binomialtree.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

using namespace std;
using namespace xf::fintech;
using namespace internal;

// test data
#include "../../data/bt_testcases.hpp"

#define STR1(x) #x
#define STR(x) STR1(x)

class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

int main(int argc, const char* argv[]) {
    // cmd parser
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    TEST_DT maxDelta = 0;
    TEST_DT tolerance = 0.02;
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    std::string data_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    if (!parser.getCmdOption("-data", data_path)) {
        std::cout << "ERROR: datafile path is not set!\n";
        return 1;
    }
    std::string inputTestCasesFileEmulationName = data_path + "/" + TestCasesFileEmulationName;
    std::string inputSVGridFileName = data_path + "/" + SVGridFileName;
    std::string inputTestCasesFileName = data_path + "/" + TestCasesFileName;

    std::map<int, std::string> optionTypeFilePath = {
        {BinomialTreeEuropeanPut, data_path + "/" + BinomialTreeEuropeanPutName + "/"},
        {BinomialTreeEuropeanCall, data_path + "/" + BinomialTreeEuropeanCallName + "/"},
        {BinomialTreeAmericanPut, data_path + "/" + BinomialTreeAmericanPutName + "/"},
        {BinomialTreeAmericanCall, data_path + "/" + BinomialTreeAmericanCallName + "/"}};

    TEST_DT npvCPUResults[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];
    cl_int err;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    // enable out of order for BINOMIAL_TREE_CU_PER_KERNEL greater that 1
    cl::CommandQueue commandQ(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                              &err);
    logger.logCreateCommandQueue(err);

    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel binomialKernel(program, BINOMIAL_TREE_KERNEL_NAME, &err);
    logger.logCreateKernel(err);
    std::vector<xf::fintech::BinomialTreeInputDataType<TEST_DT>,
                aligned_allocator<xf::fintech::BinomialTreeInputDataType<TEST_DT> > >
        inputData(BINOMIAL_TREE_MAX_OPTION_CALCULATIONS);
    std::vector<TEST_DT, aligned_allocator<TEST_DT> > npvKernelResults(BINOMIAL_TREE_MAX_OPTION_CALCULATIONS);
    size_t sizeInputDataBytes =
        sizeof(xf::fintech::BinomialTreeInputDataType<TEST_DT>) * BINOMIAL_TREE_MAX_OPTION_CALCULATIONS;
    size_t sizeOutputDataBytes = sizeof(TEST_DT) * BINOMIAL_TREE_MAX_OPTION_CALCULATIONS;

    try {
        xf::fintech::BinomialTreeInputDataType<TEST_DT> cpuInputData[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];
        int numTests = BINOMIAL_TESTCASE_NUM_V_GRID_VALUES * BINOMIAL_TESTCASE_NUM_S_GRID_VALUES;
        cl::Buffer inputDataDev(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeInputDataBytes, inputData.data());
        cl::Buffer npvKernelResultsDev(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeOutputDataBytes,
                                       npvKernelResults.data());
        cl::Event kernelEvent[BINOMIAL_TREE_CU_PER_KERNEL];
        bool verboseFlag = false;
        int narg = 1;
        TEST_DT blackScholesMaxDelta, hostMaxDelta;
        TEST_DT blackScholesAverageDelta, hostAverageDelta;

        std::vector<BinomialTestCase<TEST_DT> > testcaseData;
        BinomialTestSVGrid<TEST_DT> testSVGrid;
        BinomialTestCase<TEST_DT> tempTestCaseItem;
        std::string tmpStr;

        std::string mode = "hw";
        if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
            mode = std::getenv("XCL_EMULATION_MODE");
        }
        std::cout << "Running in " << mode << " mode" << std::endl;

        // testcase and sv grid input files
        ifstream inputFileTestCases;
        if (mode == "sw_emu" || mode == "hw_emu") {
            inputFileTestCases.open(inputTestCasesFileEmulationName);
            tolerance = 1.25; // emulation uses less tree depth => less accuracy
        } else {
            inputFileTestCases.open(inputTestCasesFileName);
            tolerance = 0.02;
        }
        ifstream inputFileSVGrid;
        inputFileSVGrid.open(inputSVGridFileName);

        // set precision to 10 decimal places
        std::cout << std::fixed;
        std::cout << std::setprecision(14);

        // skip first line as the comment
        std::getline(inputFileTestCases, tmpStr, '\n');

        while (!inputFileTestCases.eof()) {
            std::getline(inputFileTestCases, tmpStr, ',');
            if (tmpStr[0] == '#') {
                // move to the end of the line
                break;
            } else {
                tempTestCaseItem.name = tmpStr;
                std::getline(inputFileTestCases, tmpStr, ',');
                tempTestCaseItem.K = std::stod(tmpStr);
                std::getline(inputFileTestCases, tmpStr, ',');
                tempTestCaseItem.rf = std::stod(tmpStr);
                std::getline(inputFileTestCases, tmpStr, ',');
                tempTestCaseItem.T = std::stod(tmpStr);
                std::getline(inputFileTestCases, tmpStr, '\n');
                tempTestCaseItem.N = std::stoi(tmpStr);
                testcaseData.push_back(tempTestCaseItem);
            }
        }

        // debug: output the testcases parsed
        int testCaseIndex = 0;
        for (std::vector<BinomialTestCase<TEST_DT> >::iterator it = testcaseData.begin(); it != testcaseData.end();
             ++it) {
            std::cout << testCaseIndex << ": " << it->name << " " << it->K << " " << it->rf << " " << it->T << " "
                      << it->N << " " << std::endl;
            testCaseIndex++;
        }

        // skip first line as the comment
        std::getline(inputFileSVGrid, tmpStr, '\n');

        // next line contains the S grid values
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES - 1; i++) {
            std::getline(inputFileSVGrid, tmpStr, ',');
            testSVGrid.s[i] = std::stod(tmpStr);
        }
        std::getline(inputFileSVGrid, tmpStr, '\n');
        testSVGrid.s[BINOMIAL_TESTCASE_NUM_S_GRID_VALUES - 1] = std::stod(tmpStr);

        // next line contains the V grid values
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES - 1; i++) {
            std::getline(inputFileSVGrid, tmpStr, ',');
            testSVGrid.v[i] = std::stod(tmpStr);
        }
        std::getline(inputFileSVGrid, tmpStr, '\n');
        testSVGrid.v[BINOMIAL_TESTCASE_NUM_V_GRID_VALUES - 1] = std::stod(tmpStr);

        // debug: output the S & V grid parsed
        std::cout << "s grid: ";
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES; i++) {
            std::cout << testSVGrid.s[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "v grid: ";
        for (int i = 0; i < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES; i++) {
            std::cout << testSVGrid.v[i] << " ";
        }
        std::cout << std::endl;

        ///////////////////////////////////////////
        // repeat for each option type
        ///////////////////////////////////////////

        for (int k = BinomialTreeEuropeanPut; k <= BinomialTreeAmericanCall; k++) {
            std::cout << "================================================================" << std::endl;
            std::cout << "Testing:";
            switch (k) {
                case 1:
                    std::cout << BinomialTreeEuropeanPutName << std::endl;
                    break;
                case 2:
                    std::cout << BinomialTreeEuropeanCallName << std::endl;
                    break;
                case 3:
                    std::cout << BinomialTreeAmericanPutName << std::endl;
                    break;
                case 4:
                    std::cout << BinomialTreeAmericanCallName << std::endl;
                    break;
            }
            std::cout << "================================================================" << std::endl;

            // iterate around the test cases
            for (std::vector<BinomialTestCase<TEST_DT> >::iterator it = testcaseData.begin(); it != testcaseData.end();
                 ++it) {
                // loop around the s & v grid and populate
                for (int i = 0; i < BINOMIAL_TESTCASE_NUM_S_GRID_VALUES; i++) {
                    for (int j = 0; j < BINOMIAL_TESTCASE_NUM_V_GRID_VALUES; j++) {
                        int pos = (i * BINOMIAL_TESTCASE_NUM_S_GRID_VALUES) + j;

                        inputData[pos].S = testSVGrid.s[i];
                        inputData[pos].K = it->K;
                        inputData[pos].T = it->T;
                        inputData[pos].rf = it->rf;
                        inputData[pos].V = testSVGrid.v[j];
                        inputData[pos].q = 0;
                        inputData[pos].N = it->N - 1;
                        cpuInputData[pos] = inputData[pos];
                    }
                }

                std::cout << it->name << ": ";

                // CPU
                auto startHost = std::chrono::high_resolution_clock::now();
                BinomialTreeCPU(k, cpuInputData, npvCPUResults, numTests);
                auto durationHost = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::high_resolution_clock::now() - startHost)
                                        .count();

                // Kernel
                auto startKernel = std::chrono::high_resolution_clock::now();

                // Migrate input to host
                commandQ.enqueueMigrateMemObjects({inputDataDev}, 0);
                commandQ.finish();

                // Set Kernel Arguments
                narg = 0;
                binomialKernel.setArg(narg++, inputDataDev);
                binomialKernel.setArg(narg++, npvKernelResultsDev);
                binomialKernel.setArg(narg++, k);

                // Number of tests needs to be rounded up from 49 to 64 as dependant on
                // kernel PE
                binomialKernel.setArg(narg++, 64 / BINOMIAL_TREE_CU_PER_KERNEL);

                for (int i = 0; i < BINOMIAL_TREE_CU_PER_KERNEL; i++) {
                    binomialKernel.setArg(narg, i * (64 / BINOMIAL_TREE_CU_PER_KERNEL));

                    // Launch Kernel
                    commandQ.enqueueTask(binomialKernel, NULL, &kernelEvent[i]);
                }
                // Wait for all kernels to complete
                commandQ.finish();

                // Migrate output from host
                commandQ.enqueueMigrateMemObjects({npvKernelResultsDev}, CL_MIGRATE_MEM_OBJECT_HOST);
                commandQ.finish();

                // Compute overall time taken
                auto durationKernel = std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::high_resolution_clock::now() - startKernel)
                                          .count();
                // std::cout << "Kernel Duration:" << durationKernel << "us" <<
                // std::endl;

                ////////////////////////////////////////////////////////
                // Compare results from Host and Kernel
                ////////////////////////////////////////////////////////
                hostMaxDelta = 0;
                hostAverageDelta = 0;

                for (int i = 0; i < numTests; i++) {
                    TEST_DT tempHostDelta = std::abs(npvCPUResults[i] - npvKernelResults[i]);

                    if (tempHostDelta > hostMaxDelta) {
                        hostMaxDelta = tempHostDelta;
                    }

                    hostAverageDelta += pow(tempHostDelta, 2);
                }

                ////////////////////////////////////////////////////////
                // Compare results from Black Scholes Model (European)
                ////////////////////////////////////////////////////////

                ifstream inputBlackScholesFile;
                std::string blackScholesFilename = optionTypeFilePath[k] + it->name + ".txt";
                inputBlackScholesFile.open(blackScholesFilename);
                std::vector<TEST_DT> blackScholesData(BINOMIAL_TREE_MAX_OPTION_CALCULATIONS);
                for (int i = 0; i < numTests; i++) {
                    std::getline(inputBlackScholesFile, tmpStr, '\n');
                    blackScholesData[i] = std::stof(tmpStr);
                }
                inputBlackScholesFile.close();

                blackScholesMaxDelta = 0.0;
                blackScholesAverageDelta = 0.0;

                for (int i = 0; i < numTests; i++) {
                    TEST_DT tempBlackScholesDelta = std::abs(blackScholesData[i] - npvKernelResults[i]);

                    if (tempBlackScholesDelta > blackScholesMaxDelta) {
                        blackScholesMaxDelta = tempBlackScholesDelta;
                        maxDelta = tempBlackScholesDelta;
                    }

                    blackScholesAverageDelta += pow(tempBlackScholesDelta, 2);

                    // print out debug
                    verboseFlag = false;

                    if (verboseFlag) {
                        if (tempBlackScholesDelta > 0.02) {
                            // american data comes from quantlib
                            if (k >= BinomialTreeAmericanPut) {
                                std::cout << i << ": (Host)" << npvKernelResults[i] << " (FPGA)" << npvKernelResults[i]
                                          << " (Quantlib)" << blackScholesData[i] << std::endl;
                            } else {
                                std::cout << i << ": (Host)" << npvKernelResults[i] << " (FPGA)" << npvKernelResults[i]
                                          << " (BlackScholes)" << blackScholesData[i] << std::endl;
                            }
                        }
                    }
                }

                auto factor = durationHost / durationKernel;
                std::cout << "(Host) RMS:" << sqrt(hostAverageDelta / numTests) << " Max:" << hostMaxDelta
                          << " Duration:" << durationHost << "us"
                          << " (FPGA) RMS:" << sqrt(blackScholesAverageDelta / numTests)
                          << " Max:" << blackScholesMaxDelta << " Duration:" << durationKernel << "us"
                          << " (Factor:" << factor << ")" << std::endl;
            }

            std::cout << "================================================================" << std::endl;
        }

    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;

    } catch (const std::string& ex) {
        std::cout << ex << std::endl;
    } catch (...) {
        std::cout << "Exception" << std::endl;
    }

    int ret = 0;
    if (maxDelta > tolerance) {
        ret = 1;
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return ret;
}
