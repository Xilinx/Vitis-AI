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

#ifndef _XF_FINTECH_CREDIT_DEFAULT_SWAP_H_
#define _XF_FINTECH_CREDIT_DEFAULT_SWAP_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class Credit Default Swap
 *
 * @brief This class implements the Credit Default Swap Model.
 *
 * It is intended that the user will populate the parameters with
 * appropriate asset data prior to calling run() method. When the run completes
 * the number of calculated CDS spread values will be available to the user.
 */

class CreditDefaultSwap : public OCLController {
   public:
    CreditDefaultSwap(std::string xclbin_file);
    virtual ~CreditDefaultSwap();

    /**
     * Calculate one or more options based on input data and option type
     *
     * @param timesIR vector of yield curve interest rate times
     * @param ratesIR vector of yield curve interest rates rates
     * @param timesHazard vector of yield curve hazard times
     * @param ratesHazard vector of yield curve hazard rates
     * @param notional vector of notional values
     * @param recovery vector of recovery rates
     * @param maturity vector of maturities
     * @param frequency vector of frequencies
     * @param cdsSpread output vector of CDS spread values
     */
    int run(float* timesIR,
            float* ratesIR,
            float* timesHazard,
            float* ratesHazard,
            float* notional,
            float* recovery,
            float* maturity,
            int* frequency,
            float* cdsSpread);

    /**
     * This method returns the time the execution of the last call to run() took.
     */
    long long int getLastRunTime(void);

   private:
    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pCDSKernel;

    cl::Buffer* m_pHwInputRatesIR;
    cl::Buffer* m_pHwInputTimesIR;
    cl::Buffer* m_pHwInputRatesHazard;
    cl::Buffer* m_pHwInputTimesHazard;
    cl::Buffer* m_pHwInputNominal;
    cl::Buffer* m_pHwInputRecovery;
    cl::Buffer* m_pHwInputMaturity;
    cl::Buffer* m_pHwInputFrequency;
    cl::Buffer* m_pHwOutputBuffer;

    std::vector<float, aligned_allocator<float> > m_ratesIR;
    std::vector<float, aligned_allocator<float> > m_timesIR;
    std::vector<float, aligned_allocator<float> > m_ratesHazard;
    std::vector<float, aligned_allocator<float> > m_timesHazard;
    std::vector<float, aligned_allocator<float> > m_notional;
    std::vector<float, aligned_allocator<float> > m_recovery;
    std::vector<float, aligned_allocator<float> > m_maturity;
    std::vector<int, aligned_allocator<int> > m_frequency;

    std::vector<float, aligned_allocator<float> > m_hostOutputBuffer;

   private:
    std::string getKernelTypeSubString(void);
    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_CREDIT_DEFAULT_SWAP_H_ */
