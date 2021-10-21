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

#ifndef _XF_FINTECH_CF_B76_H_
#define _XF_FINTECH_CF_B76_H_

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
 * @class CFB76
 *
 * @brief This class implements the Closed Form Black 76 model.
 *
 * @details The parameter passed to the constructor controls the size of the
 * underlying buffers that will be allocated.
 * This prameter therefore controls the maximum number of assets that can be
 * processed per call to run()
 *
 * It is intended that the user will populate the input buffers with appropriate
 * asset data prior to calling run()
 * When run completes, the calculated output data will be available in the
 * relevant output buffers.
 */
class CFB76 : public OCLController {
   public:
    CFB76(unsigned int maxAssetsPerRun, std::string xclbin_file);
    virtual ~CFB76();

   public:
    /**
     * @param KDataType This is the data type that the underlying HW kernel has
     * been built with.
     *
     */
    typedef float KDataType;

   public: // INPUT BUFFERS
    KDataType* forwardPrice;
    KDataType* strikePrice;
    KDataType* volatility;
    KDataType* riskFreeRate;
    KDataType* timeToMaturity;

   public: // OUTPUT BUFFERS
    KDataType* optionPrice;
    KDataType* delta;
    KDataType* gamma;
    KDataType* vega;
    KDataType* theta;
    KDataType* rho;

   public:
    /**
     * This method is used to begin processing the asset data that is in the input
     * buffers.
     * If this function returns successfully, calculated results are available in
     * the output buffers.
     *
     * @param optionType The option type of ALL the assets data
     * @param numAssets The number of assets to process.
     */
    int run(OptionType optionType, unsigned int numAssets);

   public:
    /**
     * This method returns the time the execution of the last call to run() took
     *
     * @returns Execution time in microseconds
     */
    long long int getLastRunTime(void); // in microseconds

   protected:
    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

   protected:
    void allocateBuffers(unsigned int numRequestedElements);
    void deallocateBuffers(void);

   protected:
    std::string m_xclbin_file;
    unsigned int calculatePaddedNumElements(unsigned int numRequestedElements);
    const char* getKernelName();
    std::string getXCLBINName(Device* device);

   protected:
    unsigned int m_numPaddedBufferElements;

   private:
    static const unsigned int KERNEL_PARAMETER_BITWIDTH = 512;
    static const unsigned int NUM_ELEMENTS_PER_BUFFER_CHUNK;

   protected:
    cl::Context* m_pContext;

   private:
    cl::Program::Binaries m_binaries;

    cl::Program* m_pProgram;

   protected:
    cl::CommandQueue* m_pCommandQueue;
    cl::Kernel* m_pKernel;

   protected:
    cl::Buffer* m_pForwardPriceHWBuffer;
    cl::Buffer* m_pStrikePriceHWBuffer;
    cl::Buffer* m_pVolatilityHWBuffer;
    cl::Buffer* m_pRiskFreeRateHWBuffer;
    cl::Buffer* m_pTimeToMaturityHWBuffer;

    cl::Buffer* m_pOptionPriceHWBuffer;

    cl::Buffer* m_pDeltaHWBuffer;
    cl::Buffer* m_pGammaHWBuffer;
    cl::Buffer* m_pVegaHWBuffer;
    cl::Buffer* m_pThetaHWBuffer;
    cl::Buffer* m_pRhoHWBuffer;

   protected:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif
