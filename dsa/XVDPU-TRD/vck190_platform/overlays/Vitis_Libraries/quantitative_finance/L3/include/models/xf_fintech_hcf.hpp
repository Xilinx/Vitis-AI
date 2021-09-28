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

#ifndef _XF_FINTECH_HCF_H_
#define _XF_FINTECH_HCF_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"
// reference hcf engine in L2
#include "xf_fintech/hcf_engine.hpp"

namespace xf {
namespace fintech {

/**
 * @class hcf
 *
 * @brief This class implements the Heston Closed Form Model.
 *
 * It is intended that the user will populate the asset data structure before calling
 * the run() method and one or more calculated options will be returned.
 */

class hcf : public OCLController {
   public:
    struct hcf_input_data {
        float s0;    // stock price at t=0
        float v0;    // stock price variance at t=0
        float K;     // strike price
        float rho;   // correlation of the 2 Weiner processes
        float T;     // expiration time
        float r;     // risk free interest rate
        float kappa; // rate of reversion
        float vvol;  // volatility of volatility (sigma)
        float vbar;  // long term average variance (theta)
    };

    hcf(std::string xclbin_file);
    virtual ~hcf();

    /**
     * Calculate one or more options based on the populated inputData structure.
     *
     * @param inputData structure to be populated with the asset data
     * @param outputData one or more calculated option values returned
     * @param numOptions number of options to be calculated
     */
    int run(struct hcf_input_data* inputData, float* outputData, int numOptions);

    /**
     * Set the intergation interval width delta w.
     */
    void set_dw(int dw);

    /**
     * Set the max value of w.
     */
    void set_w_max(float w_max);

    /**
     * Get the intergation interval width delta w.
     */
    int get_dw();

    /**
     * Set the max value of w.
     */
    float get_w_max();

   private:
    static const int MAX_OPTION_CALCULATIONS = 1024;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pHcfKernel;

    cl::Buffer* m_pHwInputBuffer;
    cl::Buffer* m_pHwOutputBuffer;

    std::vector<struct xf::fintech::hcfEngineInputDataType<float>,
                aligned_allocator<struct xf::fintech::hcfEngineInputDataType<float> > >
        m_hostInputBuffer;
    std::vector<float, aligned_allocator<float> > m_hostOutputBuffer;

    int m_w_max; // the upper limit for the integration
    float m_dw;  // the delta w for the integration

    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_HCF_H_ */
