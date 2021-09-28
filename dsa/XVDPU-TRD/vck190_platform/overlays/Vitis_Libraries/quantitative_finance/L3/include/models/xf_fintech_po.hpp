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

#ifndef _XF_FINTECH_PO_H_
#define _XF_FINTECH_PO_H_

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
 * @class portfolio_optimisation
 *
 * @brief This class implements the Portfolio Optimisation Model.
 */

class portfolio_optimisation : public OCLController {
   public:
    portfolio_optimisation(std::string xclbin_file);
    virtual ~portfolio_optimisation();

    /**
     * Calculate Weights, Expected Portfolio Return and Variance for
     * Global Minimum Variance Portfolio, Tangency Portfolio, Efficient Portfolio
     * and Efficient Portfolio with risky and risk free assets
     *
     * @param inputDataFile filename of the file containing the asset prices
     * @param riskFreeRate the risk free rate
     * @param targetReturn the target return
     * @param GMVPWeights the asset weights for the Global Minimum Variance Portfolio
     * @param GMVPVariance the portfolio variance for the Global Minimum Variance Portfolio
     * @param GMVPReturn the portfolio expected return for the Global Minimum Varaince Portfolio
     * @param EffWeights the asset weights for the efficient portfolio with target return
     * @param EffVariance the portfolio variance for the efficient portfolio with target return
     * @param EffReturn the portfolio expected for the efficient portfolio with target return
     * @param TanWeights the asset weights for the Tangency Portfolio with risk free rate
     * @param TanVariance the portfolio variance for the Tangency Portfolio with risk free rate
     * @param TanReturn the portfolio expected return for the Tangency Portfolio with risk free rate
     * @param TanSharpe the Sharpe Ratio for the Tangency Portfolio with risk free rate
     * @param EffTanWeights the asset weights for the Tangency Portfolio with risk free rate and target return
     * @param EffTanVariance the portfolio variance for the Tangency Portfolio with risk free rate and target return
     * @param EffTanReturn the portfolio expected return for the Tangency Portfolio with risk free rate and target
     * return
     */
    int run(float* prices,
            int num_prices,
            int num_assets,
            float riskFreeRate,
            float targetReturn,
            std::vector<float>& GMVPWeights,
            float* GMVPVariance,
            float* GMVPReturn,
            std::vector<float>& EffWeights,
            float* EffVariance,
            float* EffReturn,
            std::vector<float>& TanWeights,
            float* TanVariance,
            float* TanReturn,
            float* TanSharpe,
            std::vector<float>& EffTanWeights,
            float* EffTanVariance,
            float* EffTanReturn);

   private:
    int max_assets = 128;
    int max_prices = 4096;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pPOKernel;

    cl::Buffer* m_pHwPricesBuffer;
    cl::Buffer* m_pHwGMVPWeightsBuffer;
    cl::Buffer* m_pHwEffWeightsBuffer;
    cl::Buffer* m_pHwTanWeightsBuffer;
    cl::Buffer* m_pHwEffTanWeightsBuffer;
    std::vector<float, aligned_allocator<float> > m_hostPrices;
    std::vector<float, aligned_allocator<float> > m_hostGMVPWeights;
    std::vector<float, aligned_allocator<float> > m_hostEffWeights;
    std::vector<float, aligned_allocator<float> > m_hostTanWeights;
    std::vector<float, aligned_allocator<float> > m_hostEffTanWeights;

    std::string m_xclbin_file;
    std::string getXCLBINName(Device* device);
};

} // end namespace fintech
} // end namespace xf

#endif /* _XF_FINTECH_PO_H_ */
