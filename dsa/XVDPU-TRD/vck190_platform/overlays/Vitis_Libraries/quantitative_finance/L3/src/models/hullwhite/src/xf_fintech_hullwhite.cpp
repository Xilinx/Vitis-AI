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

#include "models/xf_fintech_hullwhite.hpp"
#include "xf_fintech_hullwhite_kernel_constants.hpp"

using namespace xf::fintech;

HullWhiteAnalytic::HullWhiteAnalytic(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pHullWhiteKernel = nullptr;

    m_rates.clear();
    m_times.clear();
    m_currenttime.clear();
    m_maturity.clear();
    m_hostOutputBuffer.clear();

    m_pHwInputRates = nullptr;
    m_pHwInputTimes = nullptr;
    m_pHwInputCurrentTime = nullptr;
    m_pHwInputMaturity = nullptr;

    m_pHwOutputBuffer = nullptr;

    m_xclbin_file = xclbin_file;
}

HullWhiteAnalytic::~HullWhiteAnalytic() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string HullWhiteAnalytic::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int HullWhiteAnalytic::createOCLObjects(Device* device) {
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
        m_pHullWhiteKernel = new cl::Kernel(*m_pProgram, "HWA_k0", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_rates.resize(LEN);
    m_times.resize(LEN);
    m_currenttime.resize(N_k0);
    m_maturity.resize(N_k0);
    m_hostOutputBuffer.resize(N_k0);

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputRates = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * LEN,
                                         m_rates.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputTimes = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * LEN,
                                         m_times.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputCurrentTime = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                               sizeof(TEST_DT) * N_k0, m_currenttime.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputMaturity = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                            sizeof(TEST_DT) * N_k0, m_maturity.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwOutputBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                           sizeof(TEST_DT) * N_k0, m_hostOutputBuffer.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int HullWhiteAnalytic::releaseOCLObjects(void) {
    unsigned int i;

    if (m_pHwInputRates != nullptr) {
        delete (m_pHwInputRates);
        m_pHwInputRates = nullptr;
    }

    if (m_pHwInputTimes != nullptr) {
        delete (m_pHwInputTimes);
        m_pHwInputTimes = nullptr;
    }

    if (m_pHwInputCurrentTime != nullptr) {
        delete (m_pHwInputCurrentTime);
        m_pHwInputCurrentTime = nullptr;
    }

    if (m_pHwInputMaturity != nullptr) {
        delete (m_pHwInputMaturity);
        m_pHwInputMaturity = nullptr;
    }

    if (m_pHwOutputBuffer != nullptr) {
        delete (m_pHwOutputBuffer);
        m_pHwOutputBuffer = nullptr;
    }

    if (m_pHullWhiteKernel != nullptr) {
        delete (m_pHullWhiteKernel);
        m_pHullWhiteKernel = nullptr;
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

int HullWhiteAnalytic::run(double a, double sigma, double* times, double* rates, double* t, double* T, double* P) {
    int retval = XLNX_OK;

    if (deviceIsPrepared()) {
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // prepare the input data
        for (int i = 0; i < LEN; i++) {
            m_rates[i] = rates[i];
            m_times[i] = times[i];
        }

        for (int i = 0; i < N_k0; i++) {
            m_currenttime[i] = t[i];
            m_maturity[i] = T[i];
        }

        std::vector<cl::Memory> obIn0;
        obIn0.push_back(*m_pHwInputRates);
        obIn0.push_back(*m_pHwInputTimes);
        obIn0.push_back(*m_pHwInputCurrentTime);
        obIn0.push_back(*m_pHwInputMaturity);

        // output vector depedant on test case
        std::vector<cl::Memory> obOut0;
        obOut0.push_back(*m_pHwOutputBuffer);

        // Set the arguments
        m_pHullWhiteKernel->setArg(0, a);
        m_pHullWhiteKernel->setArg(1, sigma);
        m_pHullWhiteKernel->setArg(2, *m_pHwInputTimes);
        m_pHullWhiteKernel->setArg(3, *m_pHwInputRates);
        m_pHullWhiteKernel->setArg(4, *m_pHwInputCurrentTime);
        m_pHullWhiteKernel->setArg(5, *m_pHwInputMaturity);
        m_pHullWhiteKernel->setArg(6, *m_pHwOutputBuffer);

        // Copy input data to device global memory
        m_pCommandQueue->enqueueMigrateMemObjects(obIn0, 0);
        m_pCommandQueue->finish();

        // Launch the Kernel
        m_pCommandQueue->enqueueTask(*m_pHullWhiteKernel);
        m_pCommandQueue->finish();

        // Copy Result from Device Global Memory to Host Local Memory
        m_pCommandQueue->enqueueMigrateMemObjects(obOut0, CL_MIGRATE_MEM_OBJECT_HOST);
        m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        for (int i = 0; i < N_k0; i++) {
            P[i] = m_hostOutputBuffer[i];
        }
        m_runEndTime = std::chrono::high_resolution_clock::now();

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int HullWhiteAnalytic::getLastRunTime(void) {
    long long int duration = 0;
    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
    return duration;
}
