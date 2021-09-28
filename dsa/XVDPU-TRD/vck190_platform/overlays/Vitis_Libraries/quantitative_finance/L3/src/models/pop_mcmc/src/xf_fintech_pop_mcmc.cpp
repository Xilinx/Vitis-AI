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

#include "models/xf_fintech_pop_mcmc.hpp"
#include "xf_fintech_pop_mcmc_kernel_constants.hpp"

using namespace xf::fintech;

PopMCMC::PopMCMC(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pPopMCMCKernel = nullptr;

    m_hostInputBufferInv.clear();
    m_hostInputBufferSigma.clear();
    m_hostOutputBufferSamples.clear();

    mBufferInputInv = nullptr;
    mBufferInputSigma = nullptr;
    mBufferOutputSamples = nullptr;

    m_xclbin_file = xclbin_file;
}

PopMCMC::~PopMCMC() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string PopMCMC::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int PopMCMC::createOCLObjects(Device* device) {
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
        m_pPopMCMCKernel = new cl::Kernel(*m_pProgram, "mcmc_kernel", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_hostInputBufferInv.resize(NUM_CHAINS);
    m_hostInputBufferSigma.resize(NUM_CHAINS);
    m_hostOutputBufferSamples.resize(NUM_SAMPLES_MAX);

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////

    size_t sizeBufferInputInv = sizeof(double) * NUM_CHAINS;
    size_t sizeBufferInputSigma = sizeof(double) * NUM_CHAINS;
    size_t sizeBufferOutputSamples = sizeof(double) * NUM_SAMPLES_MAX;

    if (cl_retval == CL_SUCCESS) {
        mBufferInputInv = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY), sizeBufferInputInv,
                                         m_hostInputBufferInv.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        mBufferInputSigma = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY), sizeBufferInputSigma,
                                           m_hostInputBufferSigma.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        mBufferOutputSamples = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY),
                                              sizeBufferOutputSamples, m_hostOutputBufferSamples.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int PopMCMC::releaseOCLObjects(void) {
    unsigned int i;

    if (mBufferInputInv != nullptr) {
        delete (mBufferInputInv);
        mBufferInputInv = nullptr;
    }

    if (mBufferInputSigma != nullptr) {
        delete (mBufferInputSigma);
        mBufferInputSigma = nullptr;
    }

    if (mBufferOutputSamples != nullptr) {
        delete (mBufferOutputSamples);
        mBufferOutputSamples = nullptr;
    }

    if (m_pPopMCMCKernel != nullptr) {
        delete (m_pPopMCMCKernel);
        m_pPopMCMCKernel = nullptr;
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

int PopMCMC::run(int numSamples, int numBurnInSamples, double sigma, double* outputSamples) {
    int retval = XLNX_OK;

    if (deviceIsPrepared()) {
        // start time
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // prepare the data
        for (unsigned int n = 0; n < NUM_CHAINS; n++) {
            m_hostInputBufferSigma[n] = sigma;
            double temp = pow(NUM_CHAINS / (NUM_CHAINS - n), 2);
            m_hostInputBufferInv[n] = 1 / temp;
        }

        // Set the arguments
        m_pPopMCMCKernel->setArg(0, *mBufferInputInv);
        m_pPopMCMCKernel->setArg(1, *mBufferInputSigma);
        m_pPopMCMCKernel->setArg(2, *mBufferOutputSamples);
        m_pPopMCMCKernel->setArg(3, numSamples);

        // Copy input data to device global memory
        m_pCommandQueue->enqueueMigrateMemObjects({*mBufferInputInv}, 0);
        m_pCommandQueue->enqueueMigrateMemObjects({*mBufferInputSigma}, 0);

        // Launch the Kernel
        m_pCommandQueue->enqueueTask(*m_pPopMCMCKernel);

        // Copy Result from Device Global Memory to Host Local Memory
        m_pCommandQueue->enqueueMigrateMemObjects({*mBufferOutputSamples}, CL_MIGRATE_MEM_OBJECT_HOST);
        m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        for (int i = 0; i < (numSamples - numBurnInSamples); i++) {
            outputSamples[i] = m_hostOutputBufferSamples[i];
        }

        // end time
        m_runEndTime = std::chrono::high_resolution_clock::now();

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int PopMCMC::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
    return duration;
}
