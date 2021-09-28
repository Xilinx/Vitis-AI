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
#ifndef _XF_FINTECH_HJM_H_
#define _XF_FINTECH_HJM_H_

#include <cmath>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class HJM
 *
 * @brief This class implements the Heath Jarrow Morton Model.
 */
class HJM : public OCLController {
   public:
    HJM(std::string xclbin_file);
    virtual ~HJM();

    /**
     * Calculate the price of a Zero Coupon Bond using the Heath-Jarrow-Morton framework.
     *
     * @param historicalData Matrix of historical interest rate curves
     * @param noTenors Number of tenors (columns) in the historicalData matrix.
     * @param noCurves Number of interest rate curves (rows) in the historicalData matrix.
     * @param noMcPaths Number of MonteCarlo instantaneous forward rate paths to generate
     * @param simYears Number of years of forward rate curves to simulate per MC path.
     * @param zcbMaturity Maturity, in years, of the ZeroCouponBond to be priced with the HJM framework.
     * @param seeds Array of seeds for MC's RNGs. Must be 'N_FACTORS x MC_UN' elements wide.
     * @param outPrice Output price for the calculated ZeroCouponBond.
     */
    int run(double* historicalData,
            unsigned noTenors,
            unsigned noCurves,
            unsigned noMcPaths,
            float simYears,
            float zcbMaturity,
            unsigned* seeds,
            double* outPrice);

    long long getLastRunTime(void);

   private:
    // OCLCOntroller interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pHjmKernel;

    cl::Buffer* m_pHwHistDataBuffer;
    cl::Buffer* m_pHwOutPriceBuffer;
    cl::Buffer* m_pHwSeedsBuffer;

    std::vector<double, aligned_allocator<double> > m_hostHistDataBuffer;
    std::vector<double, aligned_allocator<double> > m_hostPriceDataBuffer;
    std::vector<unsigned, aligned_allocator<unsigned> > m_hostSeedsBuffer;

    std::string getKernelTypeSubString(void);

    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif // _XF_FINTECH_HJM_H_
