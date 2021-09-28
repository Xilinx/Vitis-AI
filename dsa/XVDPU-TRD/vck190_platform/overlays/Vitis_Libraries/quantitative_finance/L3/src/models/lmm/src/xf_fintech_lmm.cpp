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
#include "xf_fintech_lmm_kernel_constants.hpp"
#include "models/xf_fintech_lmm.hpp"

using namespace xf::fintech;

LMMController::LMMController(const std::string& xclbinName, const std::string& kernelName)
    : m_pContext(nullptr),
      m_pCommandQueue(nullptr),
      m_pProgram(nullptr),
      m_pKernel(nullptr),
      m_pHwPresentRateBuffer(nullptr),
      m_pHwCapletVolasBuffer(nullptr),
      m_pHwSeedsBuffer(nullptr),
      m_pHwOutPriceBuffer(nullptr),
      m_xclbinName(xclbinName),
      m_kernelName(kernelName) {}

LMMController::~LMMController() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

LMM::LMM(std::string xclbinfile)
    : m_lmmCapController(xclbinfile, "lmmCapKernel"),
      m_lmmRatchetFloaterController("lmm_ratchetfloater.xclbin", "lmmRatchetFloaterKernel"),
      m_lmmRatchetCapController("lmm_ratchetcap.xclbin", "lmmRatchetCapKernel") {}

int LMM::claimDeviceCap(Device* device) {
    return m_lmmCapController.claimDevice(device);
}

int LMM::claimDeviceRatchetCap(Device* device) {
    return m_lmmRatchetCapController.claimDevice(device);
}

int LMM::claimDeviceRatchetFloater(Device* device) {
    return m_lmmRatchetFloaterController.claimDevice(device);
}

int LMM::releaseDevice(void) {
    if (m_lmmCapController.deviceIsPrepared()) {
        return m_lmmCapController.releaseDevice();
    }
    if (m_lmmRatchetCapController.deviceIsPrepared()) {
        return m_lmmRatchetCapController.releaseDevice();
    }
    if (m_lmmRatchetFloaterController.deviceIsPrepared()) {
        return m_lmmRatchetFloaterController.releaseDevice();
    }
    return XLNX_ERROR_OCL_CONTROLLER_DOES_NOT_OWN_ANY_DEVICE;
}

int LMMController::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    cl::Device clDevice;
    clDevice = device->getCLDevice();
    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(*m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        start = std::chrono::high_resolution_clock::now();
        m_binaries.clear();
        m_binaries = xcl::import_binary_file(m_xclbinName);
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
        m_pKernel = new cl::Kernel(*m_pProgram, m_kernelName.c_str(), &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_hostPresentRateBuffer.resize(TEST_MAX_TENORS);
    m_hostCapletVolasBuffer.resize(TEST_MAX_TENORS);
    m_hostSeedsBuffer.resize(TEST_UN);
    m_hostOutPriceBuffer.resize(1);

    //////////////////////////
    // Allocate KERNEL BUFFERS
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pHwPresentRateBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(TEST_DT) * TEST_MAX_TENORS,
                           m_hostPresentRateBuffer.data(), &cl_retval);
    }
    if (cl_retval == CL_SUCCESS) {
        m_pHwCapletVolasBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(TEST_DT) * TEST_MAX_TENORS,
                           m_hostCapletVolasBuffer.data(), &cl_retval);
    }
    if (cl_retval == CL_SUCCESS) {
        m_pHwSeedsBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          sizeof(unsigned) * TEST_UN, m_hostSeedsBuffer.data(), &cl_retval);
    }
    if (cl_retval == CL_SUCCESS) {
        m_pHwOutPriceBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(TEST_DT),
                                             m_hostOutPriceBuffer.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }
    return retval;
}

int LMMController::releaseOCLObjects(void) {
    if (m_pHwPresentRateBuffer != nullptr) {
        delete m_pHwPresentRateBuffer;
        m_pHwPresentRateBuffer = nullptr;
    }
    if (m_pHwOutPriceBuffer != nullptr) {
        delete m_pHwOutPriceBuffer;
        m_pHwOutPriceBuffer = nullptr;
    }
    if (m_pHwCapletVolasBuffer != nullptr) {
        delete m_pHwCapletVolasBuffer;
        m_pHwCapletVolasBuffer = nullptr;
    }
    if (m_pHwSeedsBuffer != nullptr) {
        delete m_pHwSeedsBuffer;
        m_pHwSeedsBuffer = nullptr;
    }

    if (m_pKernel != nullptr) {
        delete m_pKernel;
        m_pKernel = nullptr;
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

int LMM::runCap(unsigned noTenors,
                unsigned noPaths,
                float* presentRate,
                float rhoBeta,
                float* capletVolas,
                float notional,
                float caprate,
                unsigned* seeds,
                float* outPrice) {
    int retval = XLNX_OK;

    if (m_lmmCapController.deviceIsPrepared()) {
        // Prepare the data
        memcpy(m_lmmCapController.m_hostSeedsBuffer.data(), seeds, TEST_UN * sizeof(unsigned));
        memcpy(m_lmmCapController.m_hostPresentRateBuffer.data(), presentRate, noTenors * sizeof(TEST_DT));
        memcpy(m_lmmCapController.m_hostCapletVolasBuffer.data(), capletVolas, (noTenors - 1) * sizeof(TEST_DT));

        // Start time
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // Set the arguments
        m_lmmCapController.m_pKernel->setArg(0, noTenors);
        m_lmmCapController.m_pKernel->setArg(1, noPaths);
        m_lmmCapController.m_pKernel->setArg(2, *m_lmmCapController.m_pHwPresentRateBuffer);
        m_lmmCapController.m_pKernel->setArg(3, rhoBeta);
        m_lmmCapController.m_pKernel->setArg(4, *m_lmmCapController.m_pHwCapletVolasBuffer);
        m_lmmCapController.m_pKernel->setArg(5, notional);
        m_lmmCapController.m_pKernel->setArg(6, caprate);
        m_lmmCapController.m_pKernel->setArg(7, *m_lmmCapController.m_pHwSeedsBuffer);
        m_lmmCapController.m_pKernel->setArg(8, *m_lmmCapController.m_pHwOutPriceBuffer);

        // Copy input data to device global memory
        m_lmmCapController.m_pCommandQueue->enqueueMigrateMemObjects(
            {*m_lmmCapController.m_pHwPresentRateBuffer, *m_lmmCapController.m_pHwCapletVolasBuffer,
             *m_lmmCapController.m_pHwSeedsBuffer},
            0);

        // Launch the kernel
        m_lmmCapController.m_pCommandQueue->enqueueTask(*m_lmmCapController.m_pKernel);

        // Copy result from device global memory to host local memory
        m_lmmCapController.m_pCommandQueue->enqueueMigrateMemObjects({*m_lmmCapController.m_pHwOutPriceBuffer},
                                                                     CL_MIGRATE_MEM_OBJECT_HOST);
        m_lmmCapController.m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        *outPrice = m_lmmCapController.m_hostOutPriceBuffer[0];

        // End time
        m_runEndTime = std::chrono::high_resolution_clock::now();
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

int LMM::runRatchetFloater(unsigned noTenors,
                           unsigned noPaths,
                           float* presentRate,
                           float rhoBeta,
                           float* capletVolas,
                           float notional,
                           float rfX,
                           float rfY,
                           float rfAlpha,
                           unsigned* seeds,
                           float* outPrice) {
    int retval = XLNX_OK;

    if (m_lmmRatchetFloaterController.deviceIsPrepared()) {
        // Prepare the data
        memcpy(m_lmmRatchetFloaterController.m_hostSeedsBuffer.data(), seeds, TEST_UN * sizeof(unsigned));
        memcpy(m_lmmRatchetFloaterController.m_hostPresentRateBuffer.data(), presentRate, noTenors * sizeof(TEST_DT));
        memcpy(m_lmmRatchetFloaterController.m_hostCapletVolasBuffer.data(), capletVolas,
               (noTenors - 1) * sizeof(TEST_DT));

        // Start time
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // Set the arguments
        m_lmmRatchetFloaterController.m_pKernel->setArg(0, noTenors);
        m_lmmRatchetFloaterController.m_pKernel->setArg(1, noPaths);
        m_lmmRatchetFloaterController.m_pKernel->setArg(2, *m_lmmRatchetFloaterController.m_pHwPresentRateBuffer);
        m_lmmRatchetFloaterController.m_pKernel->setArg(3, rhoBeta);
        m_lmmRatchetFloaterController.m_pKernel->setArg(4, *m_lmmRatchetFloaterController.m_pHwCapletVolasBuffer);
        m_lmmRatchetFloaterController.m_pKernel->setArg(5, notional);
        m_lmmRatchetFloaterController.m_pKernel->setArg(6, rfX);
        m_lmmRatchetFloaterController.m_pKernel->setArg(7, rfY);
        m_lmmRatchetFloaterController.m_pKernel->setArg(8, rfAlpha);
        m_lmmRatchetFloaterController.m_pKernel->setArg(9, *m_lmmRatchetFloaterController.m_pHwSeedsBuffer);
        m_lmmRatchetFloaterController.m_pKernel->setArg(10, *m_lmmRatchetFloaterController.m_pHwOutPriceBuffer);

        // Copy input data to device global memory
        m_lmmRatchetFloaterController.m_pCommandQueue->enqueueMigrateMemObjects(
            {*m_lmmRatchetFloaterController.m_pHwPresentRateBuffer,
             *m_lmmRatchetFloaterController.m_pHwCapletVolasBuffer, *m_lmmRatchetFloaterController.m_pHwSeedsBuffer},
            0);

        // Launch the kernel
        m_lmmRatchetFloaterController.m_pCommandQueue->enqueueTask(*m_lmmRatchetFloaterController.m_pKernel);

        // Copy result from device global memory to host local memory
        m_lmmRatchetFloaterController.m_pCommandQueue->enqueueMigrateMemObjects(
            {*m_lmmRatchetFloaterController.m_pHwOutPriceBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
        m_lmmRatchetFloaterController.m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        *outPrice = m_lmmRatchetFloaterController.m_hostOutPriceBuffer[0];

        // End time
        m_runEndTime = std::chrono::high_resolution_clock::now();
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

int LMM::runRatchetCap(unsigned noTenors,
                       unsigned noPaths,
                       float* presentRate,
                       float rhoBeta,
                       float* capletVolas,
                       float notional,
                       float spread,
                       float kappa0,
                       unsigned* seeds,
                       float* outPrice) {
    int retval = XLNX_OK;

    if (m_lmmRatchetCapController.deviceIsPrepared()) {
        // Prepare the data
        memcpy(m_lmmRatchetCapController.m_hostSeedsBuffer.data(), seeds, TEST_UN * sizeof(unsigned));
        memcpy(m_lmmRatchetCapController.m_hostPresentRateBuffer.data(), presentRate, noTenors * sizeof(TEST_DT));
        memcpy(m_lmmRatchetCapController.m_hostCapletVolasBuffer.data(), capletVolas, (noTenors - 1) * sizeof(TEST_DT));

        // Start time
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // Set the arguments
        m_lmmRatchetCapController.m_pKernel->setArg(0, noTenors);
        m_lmmRatchetCapController.m_pKernel->setArg(1, noPaths);
        m_lmmRatchetCapController.m_pKernel->setArg(2, *m_lmmRatchetCapController.m_pHwPresentRateBuffer);
        m_lmmRatchetCapController.m_pKernel->setArg(3, rhoBeta);
        m_lmmRatchetCapController.m_pKernel->setArg(4, *m_lmmRatchetCapController.m_pHwCapletVolasBuffer);
        m_lmmRatchetCapController.m_pKernel->setArg(5, notional);
        m_lmmRatchetCapController.m_pKernel->setArg(6, spread);
        m_lmmRatchetCapController.m_pKernel->setArg(7, kappa0);
        m_lmmRatchetCapController.m_pKernel->setArg(8, *m_lmmRatchetCapController.m_pHwSeedsBuffer);
        m_lmmRatchetCapController.m_pKernel->setArg(9, *m_lmmRatchetCapController.m_pHwOutPriceBuffer);

        // Copy input data to device global memory
        m_lmmRatchetCapController.m_pCommandQueue->enqueueMigrateMemObjects(
            {*m_lmmRatchetCapController.m_pHwPresentRateBuffer, *m_lmmRatchetCapController.m_pHwCapletVolasBuffer,
             *m_lmmRatchetCapController.m_pHwSeedsBuffer},
            0);

        // Launch the kernel
        m_lmmRatchetCapController.m_pCommandQueue->enqueueTask(*m_lmmRatchetCapController.m_pKernel);

        // Copy result from device global memory to host local memory
        m_lmmRatchetCapController.m_pCommandQueue->enqueueMigrateMemObjects(
            {*m_lmmRatchetCapController.m_pHwOutPriceBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
        m_lmmRatchetCapController.m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        *outPrice = m_lmmRatchetCapController.m_hostOutPriceBuffer[0];

        // End time
        m_runEndTime = std::chrono::high_resolution_clock::now();
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int LMM::getLastRunTime(void) {
    return (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
}
