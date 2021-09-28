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
#ifndef _XF_FINTECH_LMM_H_
#define _XF_FINTECH_LMM_H_

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class LMMController
 *
 * @brief This class implements the OpenCL controller for the possible pricers
 */
class LMMController : public OCLController {
    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pKernel;

    cl::Buffer* m_pHwPresentRateBuffer;
    cl::Buffer* m_pHwCapletVolasBuffer;
    cl::Buffer* m_pHwSeedsBuffer;
    cl::Buffer* m_pHwOutPriceBuffer;

    std::vector<float, aligned_allocator<float> > m_hostPresentRateBuffer;
    std::vector<float, aligned_allocator<float> > m_hostCapletVolasBuffer;
    std::vector<unsigned, aligned_allocator<unsigned> > m_hostSeedsBuffer;
    std::vector<float, aligned_allocator<float> > m_hostOutPriceBuffer;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    const std::string m_xclbinName;
    const std::string m_kernelName;

    LMMController(const std::string& xclbinName, const std::string& kernelName);
    virtual ~LMMController();

    friend class LMM;
};

/**
 * @class LMM
 *
 * @brief This class implements the LIBOR Market Model framework.
 */
class LMM {
   public:
    LMM(std::string xclbinfile);
    virtual ~LMM() {}

    /**
     * @brief Calculates the price of a cap option using the LIBOR Market Model (BGM) framework.
     *
     * @param noTenors Number of tenors in the model.
     * @param noPaths Number of MonteCarlo paths to generate.
     * @param presentRate Array with current LIBOR rates.
     * @param rhoBeta Beta parameter for correlation generation. Must be between 0 and 1.
     * @param capletVolas Implied caplet volatilities for the tenor structure, extracted with the Black76 model.
     * @param notional Notional value of the cap.
     * @param caprate Fixed caprate (K) for the cap.
     * @param seeds Array with seeds for the RNGs. Must contain @c UN seeds.
     * @param outPrice Calculated output price.
     */
    int runCap(unsigned noTenors,
               unsigned noPaths,
               float* presentRate,
               float rhoBeta,
               float* capletVolas,
               float notional,
               float caprate,
               unsigned* seeds,
               float* outPrice);

    /**
     * @brief Calculates the price of a ratchet floater option using the LIBOR Market Model (BGM) framework.
     *
     * @param noTenors Number of tenors in the model.
     * @param noPaths Number of MonteCarlo paths to generate.
     * @param presentRate Array with current LIBOR rates.
     * @param rhoBeta Beta parameter for correlation generation. Must be between 0 and 1.
     * @param capletVolas Implied caplet volatilities for the tenor structure, extracted with the Black76 model.
     * @param notional Notional value of the ratchet floater.
     * @param rfX X parameter for the ratchet floater.
     * @param rfY Y parameter for the ratchet floater.
     * @param rfAlpha alpha parameter for the ratchet floater.
     * @param seeds Array with seeds for the RNGs. Must contain @c UN seeds.
     * @param outPrice Calculated output price.
     */
    int runRatchetFloater(unsigned noTenors,
                          unsigned noPaths,
                          float* presentRate,
                          float rhoBeta,
                          float* capletVolas,
                          float notional,
                          float rfX,
                          float rfY,
                          float rfAlpha,
                          unsigned* seeds,
                          float* outPrice);

    /**
     * @brief Calculates the price of a ratchet cap option using the LIBOR Market Model (BGM) framework.
     *
     * @param noTenors Number of tenors in the model.
     * @param noPaths Number of MonteCarlo paths to generate.
     * @param presentRate Array with current LIBOR rates.
     * @param rhoBeta Beta parameter for correlation generation. Must be between 0 and 1.
     * @param capletVolas Implied caplet volatilities for the tenor structure, extracted with the Black76 model.
     * @param notional Notional value of the ratchet cap.
     * @param spread Spread parameter (s) of the ratchet cap pricing.
     * @param kappa0 Initial spread parameter (k0) of the ratchet cap pricing.
     * @param seeds Array with seeds for the RNGs. Must contain @c UN seeds.
     * @param outPrice Calculated output price.
     */
    int runRatchetCap(unsigned noTenors,
                      unsigned noPaths,
                      float* presentRate,
                      float rhoBeta,
                      float* capletVolas,
                      float notional,
                      float spread,
                      float kappa0,
                      unsigned* seeds,
                      float* outPrice);

    long long getLastRunTime(void);

    /**
     * @brief Claims the device for running Cap pricing
     */
    int claimDeviceCap(Device* device);

    /**
     * @brief Claims the device for running Ratchet Floater pricing
     */
    int claimDeviceRatchetFloater(Device* device);

    /**
     * @brief Claims the device for running Ratchet Cap pricing
     */
    int claimDeviceRatchetCap(Device* device);

    int releaseDevice(void);

   private:
    LMMController m_lmmCapController;
    LMMController m_lmmRatchetFloaterController;
    LMMController m_lmmRatchetCapController;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_runEndTime;
};

} // end namespace fintech
} // end namespace xf

#endif // _XF_FINTECH_LMM_H_
