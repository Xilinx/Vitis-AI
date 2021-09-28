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

#include "models/xf_fintech_fdbslv.hpp"

using namespace xf::fintech;

#define XSTR(X) STR(X)
#define STR(X) #X

fdbslv::fdbslv(int N, int M, std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pKernel = nullptr;

    // clear the host buffers
    m_xGrid.clear();
    m_tGrid.clear();
    m_sigma.clear();
    m_rate.clear();
    m_initialCondition.clear();
    m_solution.clear();

    m_pHwxGrid = nullptr;
    m_pHwtGrid = nullptr;
    m_pHwSigma = nullptr;
    m_pHwRate = nullptr;
    m_pHwInitialCondition = nullptr;
    m_pHwSolution = nullptr;

    m_N = N;
    m_M = M;
    m_xclbin_file = xclbin_file;
}

fdbslv::~fdbslv() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

int fdbslv::createOCLObjects(Device* device) {
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
        xclbinName = m_xclbin_file;

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
        m_pKernel = new cl::Kernel(*m_pProgram, "fd_bs_lv_kernel", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_xGrid.resize(m_N);
    m_tGrid.resize(m_M);
    m_sigma.resize(m_N * m_M);
    m_rate.resize(m_M);
    m_initialCondition.resize(m_N);
    m_solution.resize(m_N);

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////
    cl_mem_flags flags = (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY);
    size_t size = sizeof(float) * m_N;
    if (cl_retval == CL_SUCCESS) {
        m_pHwxGrid = new cl::Buffer(*m_pContext, flags, size, m_xGrid.data(), &cl_retval);
    }

    size = sizeof(float) * m_M;
    if (cl_retval == CL_SUCCESS) {
        m_pHwtGrid = new cl::Buffer(*m_pContext, flags, size, m_tGrid.data(), &cl_retval);
    }

    size = sizeof(float) * m_N * m_M;
    if (cl_retval == CL_SUCCESS) {
        m_pHwSigma = new cl::Buffer(*m_pContext, flags, size, m_sigma.data(), &cl_retval);
    }

    size = sizeof(float) * m_M;
    if (cl_retval == CL_SUCCESS) {
        m_pHwRate = new cl::Buffer(*m_pContext, flags, size, m_rate.data(), &cl_retval);
    }

    size = sizeof(float) * m_N;
    if (cl_retval == CL_SUCCESS) {
        m_pHwInitialCondition = new cl::Buffer(*m_pContext, flags, size, m_initialCondition.data(), &cl_retval);
    }

    flags = (CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY);
    size = sizeof(float) * m_N;
    if (cl_retval == CL_SUCCESS) {
        m_pHwSolution = new cl::Buffer(*m_pContext, flags, size, m_solution.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int fdbslv::releaseOCLObjects(void) {
    unsigned int i;

    // HW buffers
    if (m_pHwxGrid != nullptr) {
        delete (m_pHwxGrid);
        m_pHwxGrid = nullptr;
    }

    if (m_pHwtGrid != nullptr) {
        delete (m_pHwtGrid);
        m_pHwtGrid = nullptr;
    }

    if (m_pHwSigma != nullptr) {
        delete (m_pHwSigma);
        m_pHwSigma = nullptr;
    }

    if (m_pHwRate != nullptr) {
        delete (m_pHwRate);
        m_pHwRate = nullptr;
    }

    if (m_pHwInitialCondition != nullptr) {
        delete (m_pHwInitialCondition);
        m_pHwInitialCondition = nullptr;
    }

    if (m_pHwSolution != nullptr) {
        delete (m_pHwSolution);
        m_pHwSolution = nullptr;
    }

    // open CL objects
    if (m_pKernel != nullptr) {
        delete (m_pKernel);
        m_pKernel = nullptr;
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

int fdbslv::run(std::vector<float>& xGrid,
                std::vector<float>& tGrid,
                std::vector<float>& sigma,
                std::vector<float>& rate,
                std::vector<float>& initial_condition,
                float theta,
                float boundary_lower,
                float boundary_upper,
                std::vector<float>& solution) {
    int retval = XLNX_OK;

    if (retval == XLNX_OK) {
        if (deviceIsPrepared()) {
            // prepare the data
            for (int i = 0; i < m_N; i++) {
                m_xGrid[i] = xGrid[i];
                m_initialCondition[i] = initial_condition[i];
            }

            for (int i = 0; i < m_M; i++) {
                m_tGrid[i] = tGrid[i];
                m_rate[i] = rate[i];
            }

            for (int i = 0; i < m_N * m_M; i++) {
                m_sigma[i] = sigma[i];
            }

            // Set the arguments
            m_pKernel->setArg(0, *m_pHwxGrid);
            m_pKernel->setArg(1, *m_pHwtGrid);
            m_pKernel->setArg(2, *m_pHwSigma);
            m_pKernel->setArg(3, *m_pHwRate);
            m_pKernel->setArg(4, *m_pHwInitialCondition);
            m_pKernel->setArg(5, theta);
            m_pKernel->setArg(6, boundary_lower);
            m_pKernel->setArg(7, boundary_upper);
            m_pKernel->setArg(8, m_M);
            m_pKernel->setArg(9, *m_pHwSolution);

            // Copy input data to device global memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwxGrid}, 0);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwtGrid}, 0);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwSigma}, 0);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwRate}, 0);
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwInitialCondition}, 0);
            m_pCommandQueue->finish();

            // Launch the Kernel
            m_pCommandQueue->enqueueTask(*m_pKernel);
            m_pCommandQueue->finish();

            // Copy Result from Device Global Memory to Host Local Memory
            m_pCommandQueue->enqueueMigrateMemObjects({*m_pHwSolution}, CL_MIGRATE_MEM_OBJECT_HOST);
            m_pCommandQueue->finish();

            // --------------------------------
            // Give the caller back the results
            // --------------------------------
            for (int i = 0; i < (int)m_solution.size(); i++) {
                solution[i] = m_solution[i];
            }
        } else {
            retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
        }
    }

    return retval;
}
