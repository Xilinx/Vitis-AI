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

#ifndef _XF_FINTECH_BINOMIAL_TREE_H_
#define _XF_FINTECH_BINOMIAL_TREE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xf_fintech_device.hpp"
#include "xf_fintech_ocl_controller.hpp"
#include "xf_fintech_types.hpp"

// reference binomial tree engine in L2
#include "xf_fintech/bt_engine.hpp"

namespace xf {
namespace fintech {

/**
 * @class BinomialTree
 *
 * @brief This class implements the Binomial Tree Model.
 *
 * It is intended that the user will populate the inputData structure with
 * appropriate asset data prior to calling run() method. When the run completes
 * the calculated output data (one or more options) will be available to the user.
 */

class BinomialTree : public OCLController {
   public:
    BinomialTree(std::string xclbin_file);
    virtual ~BinomialTree();

    /**
     * Calculate one or more options based on input data and option type
     *
     * @param inputData structure to be populated with the asset data
     * @param outputData one or more calculated option values returned
     * @param optionType option type is American/European Call or Put
     * @param numOptions number of options to be calculate
     */
    int run(xf::fintech::BinomialTreeInputDataType<double>* inputData,
            double* outputData,
            int optionType,
            int numOptions);

    /**
     * This method returns the time the execution of the last call to run() took.
     */
    long long int getLastRunTime(void);

   private:
    static const int MAX_OPTION_CALCULATIONS = 1024;

    // add new kernels in here
    enum BinomialKernelType { bt_kernel_double_pe1 = 0, bt_kernel_double_pe4 = 1, bt_kernel_double_pe8 = 2 };

    // set the kernel in use
    // static const BinomialKernelType kernelInUse = bt_kernel_double_pe1;
    // static const BinomialKernelType kernelInUse = bt_kernel_double_pe4;
    // default built with PE=8
    static const BinomialKernelType kernelInUse = bt_kernel_double_pe8;

    // OCLController interface
    int createOCLObjects(Device* device);
    int releaseOCLObjects(void);

    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Program::Binaries m_binaries;
    cl::Program* m_pProgram;
    cl::Kernel* m_pBinomialKernel;

    cl::Buffer* m_pHwInputBuffer;
    cl::Buffer* m_pHwOutputBuffer;

    std::vector<xf::fintech::BinomialTreeInputDataType<double>,
                aligned_allocator<xf::fintech::BinomialTreeInputDataType<double> > >
        m_hostInputBuffer;
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

#endif /* _XF_FINTECH_BINOMIAL_TREE_H_ */
