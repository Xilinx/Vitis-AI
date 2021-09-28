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

#include "xf_fintech_error_codes.hpp"
#include "xf_fintech_trace.hpp"

#include "models/xf_fintech_po.hpp"

using namespace xf::fintech;

#define XSTR(X) STR(X)
#define STR(X) #X

portfolio_optimisation::portfolio_optimisation(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pPOKernel = nullptr;

    m_hostPrices.clear();
    m_hostGMVPWeights.clear();
    m_hostEffWeights.clear();
    m_hostTanWeights.clear();
    m_hostEffTanWeights.clear();

    m_pHwPricesBuffer = nullptr;
    m_pHwGMVPWeightsBuffer = nullptr;
    m_pHwEffWeightsBuffer = nullptr;
    m_pHwTanWeightsBuffer = nullptr;
    m_pHwEffTanWeightsBuffer = nullptr;

    m_xclbin_file = xclbin_file;
}

portfolio_optimisation::~portfolio_optimisation() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string portfolio_optimisation::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int portfolio_optimisation::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;
    std::string xclbinName;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    cl::Device clDevice;
    clDevice = device->getCLDevice();
    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(
            *m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        xclbinName = getXCLBINName(device);

        start = std::chrono::high_resolution_clock::now();
        m_binaries.clear();
        m_binaries = xcl::import_binary_file(xclbinName);
        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Binary Import Time = %lld microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    /////////////////////////
    // Create PROGRAM Object
    /////////////////////////
    if (cl_retval == CL_SUCCESS) {
        std::vector<cl::Device> devicesToProgram;
        devicesToProgram.push_back(clDevice);

        start = std::chrono::high_resolution_clock::now();
        m_pProgram = new cl::Program(*m_pContext, devicesToProgram, m_binaries, nullptr, &cl_retval);
        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Device Programming Time = %lld microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    /////////////////////////
    // Create KERNEL Objects
    /////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pPOKernel = new cl::Kernel(*m_pProgram, "po_kernel", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_hostPrices.resize(max_assets * max_prices);
    m_hostGMVPWeights.resize(max_assets + 2); // +2 for portfolio return and variance
    m_hostEffWeights.resize(max_assets + 2);
    m_hostTanWeights.resize(max_assets + 3); // +3 for the above and the Sharpe Ratio
    m_hostEffTanWeights.resize(max_assets);

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////
    size_t size = sizeof(float) * max_assets * max_prices;
    if (cl_retval == CL_SUCCESS) {
        m_pHwPricesBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY), size,
                                           m_hostPrices.data(), &cl_retval);
    }

    size = sizeof(float) * (max_assets + 2);
    if (cl_retval == CL_SUCCESS) {
        m_pHwGMVPWeightsBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY), size,
                                                m_hostGMVPWeights.data(), &cl_retval);
    }

    size = sizeof(float) * (max_assets + 2);
    if (cl_retval == CL_SUCCESS) {
        m_pHwEffWeightsBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY), size,
                                               m_hostEffWeights.data(), &cl_retval);
    }

    size = sizeof(float) * (max_assets + 3);
    if (cl_retval == CL_SUCCESS) {
        m_pHwTanWeightsBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY), size,
                                               m_hostTanWeights.data(), &cl_retval);
    }

    size = sizeof(float) * (max_assets);
    if (cl_retval == CL_SUCCESS) {
        m_pHwEffTanWeightsBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY), size,
                                                  m_hostEffTanWeights.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int portfolio_optimisation::releaseOCLObjects(void) {
    unsigned int i;

    if (m_pHwPricesBuffer != nullptr) {
        delete (m_pHwPricesBuffer);
        m_pHwPricesBuffer = nullptr;
    }
    if (m_pHwGMVPWeightsBuffer != nullptr) {
        delete (m_pHwGMVPWeightsBuffer);
        m_pHwGMVPWeightsBuffer = nullptr;
    }
    if (m_pHwEffWeightsBuffer != nullptr) {
        delete (m_pHwEffWeightsBuffer);
        m_pHwEffWeightsBuffer = nullptr;
    }
    if (m_pHwTanWeightsBuffer != nullptr) {
        delete (m_pHwTanWeightsBuffer);
        m_pHwTanWeightsBuffer = nullptr;
    }
    if (m_pHwEffTanWeightsBuffer != nullptr) {
        delete (m_pHwEffTanWeightsBuffer);
        m_pHwEffTanWeightsBuffer = nullptr;
    }

    if (m_pPOKernel != nullptr) {
        delete (m_pPOKernel);
        m_pPOKernel = nullptr;
    }

    if (m_pProgram != nullptr) {
        delete (m_pProgram);
        m_pProgram = nullptr;
    }

    for (i = 0; i < m_binaries.size(); i++) {
        std::pair<const void*, cl::size_type> binaryPair = m_binaries[i];
        delete[](char*)(binaryPair.first);
    }

    if (m_pCommandQueue != nullptr) {
        delete (m_pCommandQueue);
        m_pCommandQueue = nullptr;
    }

    if (m_pContext != nullptr) {
        delete (m_pContext);
        m_pContext = nullptr;
    }

    return 0;
}

int portfolio_optimisation::run(float* prices,
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
                                float* EffTanReturn) {
    int retval = XLNX_OK;

    if (retval == XLNX_OK) {
        if (deviceIsPrepared()) {
            // prepare the data
            for (int i = 0; i < num_assets * num_prices; i++) {
                m_hostPrices[i] = prices[(i % num_assets) * num_prices + i / num_assets];
            }

            // Set the arguments
            m_pPOKernel->setArg(0, *m_pHwPricesBuffer);
            m_pPOKernel->setArg(1, num_assets);
            m_pPOKernel->setArg(2, num_prices);
            m_pPOKernel->setArg(3, targetReturn);
            m_pPOKernel->setArg(4, riskFreeRate);
            m_pPOKernel->setArg(5, *m_pHwGMVPWeightsBuffer);
            m_pPOKernel->setArg(6, *m_pHwEffWeightsBuffer);
            m_pPOKernel->setArg(7, *m_pHwTanWeightsBuffer);
            m_pPOKernel->setArg(8, *m_pHwEffTanWeightsBuffer);

            // Copy input data to device global memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwPricesBuffer}, 0);
            m_pCommandQueue->finish();

            // Launch the Kernel
            m_pCommandQueue->enqueueTask(*m_pPOKernel);
            m_pCommandQueue->finish();

            // Copy Result from Device Global Memory to Host Local Memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwGMVPWeightsBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwEffWeightsBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwTanWeightsBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwEffTanWeightsBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->finish();

            // --------------------------------
            // Give the caller back the results
            // --------------------------------
            int i;
            for (i = 0; i < num_assets; i++) {
                GMVPWeights[i] = m_hostGMVPWeights[i];
                EffWeights[i] = m_hostEffWeights[i];
                TanWeights[i] = m_hostTanWeights[i];
                EffTanWeights[i] = m_hostEffTanWeights[i];
            }
            *GMVPReturn = m_hostGMVPWeights[i];
            *EffReturn = m_hostEffWeights[i];
            *TanReturn = m_hostTanWeights[i];
            i++;
            *GMVPVariance = m_hostGMVPWeights[i];
            *EffVariance = m_hostEffWeights[i];
            *TanVariance = m_hostTanWeights[i];
            i++;
            *TanSharpe = m_hostTanWeights[i];

        } else {
            retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
        }
    }

    return retval;
}
