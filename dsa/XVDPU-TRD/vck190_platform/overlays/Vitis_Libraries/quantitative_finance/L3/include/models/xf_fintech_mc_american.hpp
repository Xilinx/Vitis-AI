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

#ifndef _XF_FINTECH_MC_AMERICAN_H_
#define _XF_FINTECH_MC_AMERICAN_H_

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class MCAmerican
 *
 * @brief This class implements the Monte-Carlo American model.
 *
 */
class MCAmerican : public OCLController {
   public:
    MCAmerican();
    virtual ~MCAmerican();

   public:
    /**
     * Run a single asset until the REQUIRED TOLERANCE is met...
     *
     * @param optionType either American/European Call or Put
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRate the risk free interest rate
     * @param dividendYield the dividend yield
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param requiredTolerance the tolerance
     * @param pOptionPrice the returned option price
     *
     */
    int run(OptionType optionType,
            double stockPrice,
            double strikePrice,
            double riskFreeRate,
            double dividendYield,
            double volatility,
            double timeToMaturity,
            double requiredTolerance,
            double* pOptionPrice);

    /**
     * Run a single asset for the REQUIRED NUMBER OF SAMPLES...
     *
     * @param optionType either American/European Call or Put
     * @param stockPrice the stock price
     * @param strikePrice the strike price
     * @param riskFreeRate the risk free interest rate
     * @param dividendYield the dividend yield
     * @param volatility the volatility
     * @param timeToMaturity the time to maturity
     * @param requiredSamples the number of samples
     * @param pOptionPrice the returned option price
     *
     */
    int run(OptionType optionType,
            double stockPrice,
            double strikePrice,
            double riskFreeRate,
            double dividendYield,
            double volatility,
            double timeToMaturity,
            unsigned int requiredSamples,
            double* pOptionPrice);

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
    int runInternal(OptionType optionType,
                    double stockPrice,
                    double strikePrice,
                    double riskFreeRate,
                    double dividendYield,
                    double volatility,
                    double timeToMaturity,
                    double requiredTolerance,
                    unsigned int requiredSamples,
                    double* pOptionPrice);

   private:
    std::string getXCLBINName(Device* device);

   private:
    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;

    cl::Program::Binaries m_binaries;

    cl::Program* m_pProgram;

    uint8_t* m_hostOutputPricesBuffer;
    uint8_t* m_hostOutputMatrixBuffer;
    uint8_t* m_hostCoeffBuffer;
    void* m_hostOutputBuffer1;
    void* m_hostOutputBuffer2;

    cl::Kernel* m_pPreSampleKernel;
    cl::Kernel* m_pCalibrationKernel;
    cl::Kernel* m_pPricingKernel1;
    cl::Kernel* m_pPricingKernel2;

    cl_mem_ext_ptr_t m_outputPriceBufferOptions;
    cl_mem_ext_ptr_t m_outputMatrixBufferOptions;
    cl_mem_ext_ptr_t m_coeffBufferOptions;
    cl_mem_ext_ptr_t m_outputBufferOptions1;
    cl_mem_ext_ptr_t m_outputBufferOptions2;

    cl::Buffer* m_pHWOutputPriceBuffer;
    cl::Buffer* m_pHWOutputMatrixBuffer;
    cl::Buffer* m_pHWCoeffBuffer;
    cl::Buffer* m_pHWOutputBuffer1;
    cl::Buffer* m_pHWOutputBuffer2;

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif
