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

#ifndef _XF_FINTECH_FDBSLV_H_
#define _XF_FINTECH_FDBSLV_H_

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
 * @class fdbslv
 *
 * @brief This class implements the Finite Difference Black Scholes Local Volatility Model.
 */

class fdbslv : public OCLController {
   public:
    fdbslv(int N, int M, std::string xclbin_file);
    virtual ~fdbslv();

    /**
     * Calculate the pricing grid based on the input data
     *
     * TODO
     */
    int run(std::vector<float>& xGrid,
            std::vector<float>& tGrid,
            std::vector<float>& sigma,
            std::vector<float>& rate,
            std::vector<float>& initial_conditions,
            float theta,
            float boundary_lower,
            float boundary_upper,
            std::vector<float>& solution);

   private:
    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pKernel;

    int m_N;
    int m_M;

    // host buffers
    std::vector<float, aligned_allocator<float> > m_xGrid;
    std::vector<float, aligned_allocator<float> > m_tGrid;
    std::vector<float, aligned_allocator<float> > m_sigma;
    std::vector<float, aligned_allocator<float> > m_rate;
    std::vector<float, aligned_allocator<float> > m_initialCondition;
    std::vector<float, aligned_allocator<float> > m_solution;

    // cl buffers
    cl::Buffer* m_pHwxGrid;
    cl::Buffer* m_pHwtGrid;
    cl::Buffer* m_pHwSigma;
    cl::Buffer* m_pHwRate;
    cl::Buffer* m_pHwInitialCondition;
    cl::Buffer* m_pHwSolution;

    std::string m_xclbin_file;
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_FDBSLV_H_ */
