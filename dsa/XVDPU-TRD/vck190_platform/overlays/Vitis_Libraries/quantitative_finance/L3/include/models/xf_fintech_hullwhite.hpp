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

#ifndef _XF_FINTECH_HULLWHITE_H_
#define _XF_FINTECH_HULLWHITE_H_

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
 * @class Hull White Analytic
 *
 * @brief This class implements the Credit Default Swap Model.
 *
 * It is intended that the user will populate the parameters with
 * appropriate asset data prior to calling run() method. When the run completes
 * the number of calculated CDS spread values will be available to the user.
 */

class HullWhiteAnalytic : public OCLController {
   public:
    HullWhiteAnalytic(std::string xclbin_file);
    virtual ~HullWhiteAnalytic();

    /**
     * Calculate one or more options based on input data and option type
     *
     * @param a The mean reversion
     * @param sigma The volatility
     * @param times A vector of maturity's
     * @param rates A vector of interest rates
     * @param t A vector of current time t
     * @param T A vector of maturity time T
     * @param P  Output vector of bond prices
     */
    int run(double a, double sigma, double* times, double* rates, double* t, double* T, double* P);

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
    cl::Kernel* m_pHullWhiteKernel;

    cl::Buffer* m_pHwInputRates;
    cl::Buffer* m_pHwInputTimes;
    cl::Buffer* m_pHwInputCurrentTime;
    cl::Buffer* m_pHwInputMaturity;
    cl::Buffer* m_pHwOutputBuffer;

    std::vector<double, aligned_allocator<double> > m_rates;
    std::vector<double, aligned_allocator<double> > m_times;
    std::vector<double, aligned_allocator<double> > m_currenttime;
    std::vector<double, aligned_allocator<double> > m_maturity;
    std::vector<double, aligned_allocator<double> > m_hostOutputBuffer;

   private:
    std::string getKernelTypeSubString(void);
    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_HULLWHITE_H_ */
