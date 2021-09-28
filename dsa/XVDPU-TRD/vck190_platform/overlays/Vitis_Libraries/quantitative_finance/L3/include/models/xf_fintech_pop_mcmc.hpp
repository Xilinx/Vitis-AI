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

#ifndef _XF_FINTECH_POP_MCMC_H_
#define _XF_FINTECH_POP_MCMC_H_

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
 * @class PopMCMC
 *
 * @brief This class implements the Monte Carlo Markov Chain Model.
 *
 * The user calls the run() method passing in the number of samples to be
 * generated, the number to be discarded and a sigma value, the method then
 * returns the generated values.
 */

class PopMCMC : public OCLController {
   public:
    PopMCMC(std::string xclbin_file);
    virtual ~PopMCMC();

    /**
     * Generate a number of samples.
     *
     * @param numSamples the number of samples to generate.
     * @param numBurnInSamples the number samples to discard at the start.
     * @param sigma the sigma value.
     * @param outputData the generated data.
     */
    int run(int numSamples, int numBurnInSamples, double sigma, double* outputData);

    /**
     * This method returns the time the execution of the last call to run() took.
     */
    long long int getLastRunTime(void);

   private:
    static const int NUM_CHAINS = 10;
    static const int NUM_SAMPLES_MAX = 5000;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pPopMCMCKernel;

    cl::Buffer* mBufferInputInv;
    cl::Buffer* mBufferInputSigma;
    cl::Buffer* mBufferOutputSamples;

    std::vector<double, aligned_allocator<double> > m_hostInputBufferInv;
    std::vector<double, aligned_allocator<double> > m_hostInputBufferSigma;
    std::vector<double, aligned_allocator<double> > m_hostOutputBufferSamples;

    std::string getKernelTypeSubString(void);
    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech

} // end namespace xf

#endif /* _XF_FINTECH_POP_MCMC_H_ */
