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

#ifndef _XF_FINTECH_CF_BLACK_SCHOLES_MERTON_H_
#define _XF_FINTECH_CF_BLACK_SCHOLES_MERTON_H_

#include "xf_fintech_cf_black_scholes.hpp"

namespace xf {
namespace fintech {

/**
 * @class CFBlackScholesMerton
 *
 * @brief This class implements the Closed Form Black Scholes Merton model.
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
class CFBlackScholesMerton : public CFBlackScholes {
   public:
    CFBlackScholesMerton(unsigned int maxAssetsPerRun, std::string xclbin_file);
    virtual ~CFBlackScholesMerton();

   public: // INPUT BUFFER
    KDataType* dividendYield;

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

   private:
    const char* getKernelName(void);
    int releaseOCLObjects(void);
    int createOCLObjects(Device* device);
    void allocateBuffers(unsigned int numRequestedElements);
    cl::Buffer* m_pDividendYieldHWBuffer;
};

} // end namespace fintech
} // end namespace xf

#endif
