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

#include "models/xf_fintech_binomialtree.hpp"
#include "xf_fintech_binomialtree_kernel_constants.hpp"

using namespace xf::fintech;

BinomialTree::BinomialTree(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pBinomialKernel = nullptr;

    m_hostInputBuffer.clear();
    m_hostOutputBuffer.clear();

    m_pHwInputBuffer = nullptr;
    m_pHwOutputBuffer = nullptr;

    m_xclbin_file = xclbin_file;
}

BinomialTree::~BinomialTree() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string BinomialTree::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int BinomialTree::createOCLObjects(Device* device) {
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
        m_pBinomialKernel = new cl::Kernel(*m_pProgram, "BinomialTreeKernel", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_hostInputBuffer.resize(MAX_OPTION_CALCULATIONS);
    m_hostOutputBuffer.resize(MAX_OPTION_CALCULATIONS);

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////

    size_t sizeInputBufferBytes = sizeof(BinomialTreeInputDataType<TEST_DT>) * MAX_OPTION_CALCULATIONS;
    size_t sizeOuputBufferBytes = sizeof(TEST_DT) * MAX_OPTION_CALCULATIONS;

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY), sizeInputBufferBytes,
                                          m_hostInputBuffer.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwOutputBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY), sizeOuputBufferBytes,
                                           m_hostOutputBuffer.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int BinomialTree::releaseOCLObjects(void) {
    unsigned int i;

    if (m_pHwInputBuffer != nullptr) {
        delete (m_pHwInputBuffer);
        m_pHwInputBuffer = nullptr;
    }

    if (m_pHwOutputBuffer != nullptr) {
        delete (m_pHwOutputBuffer);
        m_pHwOutputBuffer = nullptr;
    }

    if (m_pBinomialKernel != nullptr) {
        delete (m_pBinomialKernel);
        m_pBinomialKernel = nullptr;
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

int BinomialTree::run(xf::fintech::BinomialTreeInputDataType<double>* inputData,
                      double* outputData,
                      int optionType,
                      int numOptions) {
    int retval = XLNX_OK;

    if (kernelInUse == bt_kernel_double_pe4) {
        if (numOptions % 4 != 0) {
            Trace::printError("[XLNX] BinomialTree::run - number of options to calculate should be a multiple of 4\n");
            retval = XLNX_ERROR_MODEL_INTERNAL_ERROR;
        }

    } else if (kernelInUse == bt_kernel_double_pe8) {
        if (numOptions % 8 != 0) {
            Trace::printError("[XLNX] BinomialTree::run - number of options to calculate should be a multiple of 8\n");
            retval = XLNX_ERROR_MODEL_INTERNAL_ERROR;
        }
    }

    if (retval == XLNX_OK) {
        if (deviceIsPrepared()) {
            int num_options = numOptions;
            int start_index = 0;
            int option_type = optionType;

            m_runStartTime = std::chrono::high_resolution_clock::now();

            // prepare the data
            for (int i = 0; i < numOptions; i++) {
                m_hostInputBuffer[i].S = inputData[i].S;
                m_hostInputBuffer[i].K = inputData[i].K;
                m_hostInputBuffer[i].T = inputData[i].T;
                m_hostInputBuffer[i].rf = inputData[i].rf;
                m_hostInputBuffer[i].V = inputData[i].V;
                m_hostInputBuffer[i].q = inputData[i].q;
                m_hostInputBuffer[i].N = inputData[i].N;
            }

            // Set the arguments
            m_pBinomialKernel->setArg(0, *m_pHwInputBuffer);
            m_pBinomialKernel->setArg(1, *m_pHwOutputBuffer);
            m_pBinomialKernel->setArg(2, option_type);
            m_pBinomialKernel->setArg(3, num_options);
            m_pBinomialKernel->setArg(4, start_index);

            // Copy input data to device global memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwInputBuffer}, 0);
            m_pCommandQueue->finish();

            // Launch the Kernel
            m_pCommandQueue->enqueueTask(*m_pBinomialKernel);
            m_pCommandQueue->finish();

            // Copy Result from Device Global Memory to Host Local Memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwOutputBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->finish();

            // --------------------------------
            // Give the caller back the results
            // --------------------------------
            for (int i = 0; i < numOptions; i++) {
                outputData[i] = m_hostOutputBuffer[i];
            }

            m_runEndTime = std::chrono::high_resolution_clock::now();
        } else {
            retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
        }
    }

    return retval;
}

long long int BinomialTree::getLastRunTime(void) {
    long long int duration = 0;
    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
    return duration;
}
