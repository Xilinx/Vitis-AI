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

#ifndef _XF_FINTECH_M76_H_
#define _XF_FINTECH_M76_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"
#include "xf_fintech/m76_engine_defn.hpp" // struct jump_diffusion_params

namespace xf {
namespace fintech {

/**
 * @class m76
 *
 * @brief This class implements the Merton 76 Jump Diffusion Model.
 *
 * It is intended that the user will populate the asset data when calling the run() method.
 */

class m76 : public OCLController {
   public:
    struct m76_input_data {
        float S;      // stock price at t=0
        float sigma;  // stock price volatility
        float K;      // strike price
        float r;      // risk free interest rate
        float T;      // time to vest (years)
        float lambda; // mean jump per unit time
        float kappa;  // expected[Y-1] Y is the random variable
        float delta;  // root of variance of ln(Y)
    };

    m76(std::string xclbin_file);
    virtual ~m76();

    /**
     * Calculate one or more options based on input data and option type
     *
     * @param inputData structure to be populated with the asset data
     * @param outputData one or more calculated option values returned
     * @param numOptions number of options to be calculated
     */
    int run(struct m76_input_data* inputData, float* outputData, int numOptions);

   private:
    static const int MAX_OPTION_CALCULATIONS = 2048;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pM76Kernel;

    cl::Buffer* m_pHwInputBuffer;
    cl::Buffer* m_pHwOutputBuffer;
    std::vector<struct xf::fintech::jump_diffusion_params<float>,
                aligned_allocator<struct xf::fintech::jump_diffusion_params<float> > >
        m_hostInputBuffer;
    std::vector<float, aligned_allocator<float> > m_hostOutputBuffer;

    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_M76_H_ */
