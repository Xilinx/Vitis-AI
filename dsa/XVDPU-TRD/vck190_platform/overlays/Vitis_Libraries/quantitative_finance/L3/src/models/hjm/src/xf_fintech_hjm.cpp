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
#include "xf_fintech_hjm_kernel_constants.hpp"
#include "models/xf_fintech_hjm.hpp"

#define KERNEL_NAME "hjm_kernel"

using namespace xf::fintech;

HJM::HJM(std::string xclbin_file)
    : m_pContext(nullptr),
      m_pCommandQueue(nullptr),
      m_pProgram(nullptr),
      m_pHjmKernel(nullptr),
      m_pHwHistDataBuffer(nullptr),
      m_pHwOutPriceBuffer(nullptr),
      m_pHwSeedsBuffer(nullptr) {
    m_xclbin_file = xclbin_file;
}

HJM::~HJM() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string HJM::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int HJM::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;
    std::string xclbinName;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    cl::Device clDevice;
    clDevice = device->getCLDevice();
    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(*m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        xclbinName = getXCLBINName(device);

        start = std::chrono::high_resolution_clock::now();
        m_binaries.clear();
        m_binaries = xcl::import_binary_file(xclbinName);
        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Binary Import Time = %ldd microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    //////////////////////////
    // Create PROGRAM object
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        std::vector<cl::Device> devicesToProgram;
        devicesToProgram.push_back(clDevice);

        start = std::chrono::high_resolution_clock::now();
        m_pProgram = new cl::Program(*m_pContext, devicesToProgram, m_binaries, nullptr, &cl_retval);
        end = std::chrono::high_resolution_clock::now();

        Trace::printInfo("[XLNX] Device Programming Time = %ldd microseconds\n",
                         std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    //////////////////////////
    // Create KERNEL Objects
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pHjmKernel = new cl::Kernel(*m_pProgram, KERNEL_NAME, &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_hostHistDataBuffer.resize(MAX_TENORS * MAX_CURVES);
    m_hostSeedsBuffer.resize(MC_UN * N_FACTORS);
    m_hostPriceDataBuffer.resize(1);

    //////////////////////////
    // Allocate KERNEL BUFFERS
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pHwHistDataBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           sizeof(KDataType) * MAX_TENORS * MAX_CURVES, m_hostHistDataBuffer.data(), &cl_retval);
    }
    if (cl_retval == CL_SUCCESS) {
        m_pHwOutPriceBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(KDataType),
                                             m_hostPriceDataBuffer.data(), &cl_retval);
    }
    if (cl_retval == CL_SUCCESS) {
        m_pHwSeedsBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          MC_UN * N_FACTORS * sizeof(unsigned), m_hostSeedsBuffer.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }
    return retval;
}

int HJM::releaseOCLObjects(void) {
    if (m_pHwHistDataBuffer != nullptr) {
        delete m_pHwHistDataBuffer;
        m_pHwHistDataBuffer = nullptr;
    }
    if (m_pHwOutPriceBuffer != nullptr) {
        delete m_pHwOutPriceBuffer;
        m_pHwOutPriceBuffer = nullptr;
    }
    if (m_pHwSeedsBuffer != nullptr) {
        delete m_pHwSeedsBuffer;
        m_pHwSeedsBuffer = nullptr;
    }

    if (m_pHjmKernel != nullptr) {
        delete m_pHjmKernel;
        m_pHjmKernel = nullptr;
    }
    if (m_pProgram != nullptr) {
        delete m_pProgram;
        m_pProgram = nullptr;
    }

    for (unsigned i = 0; i < m_binaries.size(); i++) {
        auto binaryPair = m_binaries[i];
        delete[](char*)(binaryPair.first);
    }

    if (m_pCommandQueue != nullptr) {
        delete m_pCommandQueue;
        m_pCommandQueue = nullptr;
    }
    if (m_pContext != nullptr) {
        delete m_pContext;
        m_pContext = nullptr;
    }
    return 0;
}

int HJM::run(double* historicalData,
             unsigned noTenors,
             unsigned noCurves,
             unsigned noMcPaths,
             float simYears,
             float zcbMaturity,
             unsigned* seeds,
             double* outputPrice) {
    int retval = XLNX_OK;

    if (deviceIsPrepared()) {
        // Prepare the data
        memcpy(m_hostSeedsBuffer.data(), seeds, MC_UN * N_FACTORS * sizeof(unsigned));
        memcpy(m_hostHistDataBuffer.data(), historicalData, noTenors * noCurves * sizeof(double));

        // Start time
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // Set the arguments
        m_pHjmKernel->setArg(0, *m_pHwHistDataBuffer);
        m_pHjmKernel->setArg(1, noTenors);
        m_pHjmKernel->setArg(2, noCurves);
        m_pHjmKernel->setArg(3, simYears);
        m_pHjmKernel->setArg(4, noMcPaths);
        m_pHjmKernel->setArg(5, zcbMaturity);
        m_pHjmKernel->setArg(6, *m_pHwSeedsBuffer);
        m_pHjmKernel->setArg(7, *m_pHwOutPriceBuffer);

        // Copy input data to device global memory
        m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwHistDataBuffer, *m_pHwSeedsBuffer}, 0);

        // Launch the kernel
        m_pCommandQueue->enqueueTask(*m_pHjmKernel);

        // Copy result from device global memory to host local memory
        m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwOutPriceBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
        m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        *outputPrice = m_hostPriceDataBuffer[0];

        // End time
        m_runEndTime = std::chrono::high_resolution_clock::now();
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int HJM::getLastRunTime(void) {
    return (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
}
