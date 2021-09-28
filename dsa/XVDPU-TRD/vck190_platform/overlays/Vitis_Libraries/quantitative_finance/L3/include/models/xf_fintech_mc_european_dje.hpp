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
#ifndef _XF_FINTECH_MC_EUROPEAN_DJE_H_
#define _XF_FINTECH_MC_EUROPEAN_DJE_H_

#include <chrono>
#include <string>

#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class MCEuropeanDJE
 *
 * @brief The MCEuropeanDJE (Dow Jones Engine) class takes in an array of asset
 * data and calculates a single Dow Jones Industrial Average (DJIA) value.
 *
 */
class MCEuropeanDJE : public OCLController {
   public:
    MCEuropeanDJE(std::string xclbin_file);
    virtual ~MCEuropeanDJE();

   public:
    /**
     * The following constant defines the number of KERNELS that have been built
     * in the HW...
     */
    static const int NUM_KERNELS = 1;

    /**
     * Process arrays of asset data, until required TOLERANCE is met...
     *
     * @param optionType either American/European Call or Put
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRate the risk free interest rate
     * @param dividendYield the dividend yield
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param requiredTolerance the required tolerance
     * @param numAssets the number of assets
     * @param dowDivisor the Dow Divisor
     * @param DJIAOutput the returned Dow Jones Industrial Average
     *
     */
    int run(OptionType* optionType,
            double* stockPrice,
            double* strikePrice,
            double* riskFreeRate,
            double* dividendYield,
            double* volatility,
            double* timeToMaturity,
            double* requiredTolerance,
            unsigned int numAssets,
            double dowDivisor,
            double* DJIAOutput);

    /**
     * Process arrays of asset data, for REQUIRED NUMBER OF SAMPLES...
     *
     * @param optionType either American/European Call or Put
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRate the risk free interest rate
     * @param dividendYield the dividend yield
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param requiredSamples the required samples
     * @param numAssets the number of assets
     * @param dowDivisor the Dow Divisor
     * @param DJIAOutput the returned Dow Jones Industrial Average
     *
     */
    int run(OptionType* optionType,
            double* stockPrice,
            double* strikePrice,
            double* riskFreeRate,
            double* dividendYield,
            double* volatility,
            double* timeToMaturity,
            unsigned int* requiredSamples,
            unsigned int numAssets,
            double dowDivisor,
            double* DJIAOutput);

   public:
    /**
     * This method returns the time the execution of the last call to run() took
     *
     * @returns Execution time in microseconds
     */
    long long int getLastRunTime(void);

   private:
    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

   private:
    // Run multiple asset values...
    int runInternal(OptionType* optionType,
                    double* stockPrice,
                    double* strikePrice,
                    double* riskFreeRate,
                    double* dividendYield,
                    double* volatility,
                    double* timeToMaturity,
                    double* requiredTolerance,
                    unsigned int* requiredSamples,
                    double* optionValues,
                    unsigned int numAssets);

   private:
    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);

   private:
    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;

    cl::Program::Binaries m_binaries;

    cl::Program* m_pProgram;

    static const char* KERNEL_NAMES[NUM_KERNELS];

    cl::Kernel* m_pKernels[NUM_KERNELS];

    void* m_hostOutputBuffers[NUM_KERNELS];

    cl_mem_ext_ptr_t m_hwBufferOptions[NUM_KERNELS];

    cl::Buffer* m_pHWBuffers[NUM_KERNELS];

   private:
   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif //_XF_FINTECH_MC_EUROPEAN_DJE_H_
