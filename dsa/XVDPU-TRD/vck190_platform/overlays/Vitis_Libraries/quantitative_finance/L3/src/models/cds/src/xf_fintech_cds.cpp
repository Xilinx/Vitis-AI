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

#include "models/xf_fintech_cds.hpp"
#include "xf_fintech_cds_kernel_constants.hpp"

using namespace xf::fintech;

CreditDefaultSwap::CreditDefaultSwap(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pCDSKernel = nullptr;

    m_ratesIR.clear();
    m_timesIR.clear();
    m_ratesHazard.clear();
    m_timesHazard.clear();
    m_notional.clear();
    m_recovery.clear();
    m_maturity.clear();
    m_frequency.clear();

    m_hostOutputBuffer.clear();

    m_pHwInputRatesIR = nullptr;
    m_pHwInputTimesIR = nullptr;
    m_pHwInputRatesHazard = nullptr;
    m_pHwInputTimesHazard = nullptr;
    m_pHwInputNominal = nullptr;
    m_pHwInputRecovery = nullptr;
    m_pHwInputMaturity = nullptr;
    m_pHwInputFrequency = nullptr;

    m_pHwOutputBuffer = nullptr;

    m_xclbin_file = xclbin_file;
}

CreditDefaultSwap::~CreditDefaultSwap() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string CreditDefaultSwap::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int CreditDefaultSwap::createOCLObjects(Device* device) {
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
        m_pCDSKernel = new cl::Kernel(*m_pProgram, "CDS_kernel", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    m_ratesIR.resize(IRLEN);
    m_timesIR.resize(IRLEN);
    m_ratesHazard.resize(HAZARDLEN);
    m_timesHazard.resize(HAZARDLEN);
    m_notional.resize(N);
    m_recovery.resize(N);
    m_maturity.resize(N);
    m_frequency.resize(N);
    m_hostOutputBuffer.resize(N);

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputRatesIR = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                           sizeof(TEST_DT) * IRLEN, m_ratesIR.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputTimesIR = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                           sizeof(TEST_DT) * IRLEN, m_timesIR.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputRatesHazard = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                               sizeof(TEST_DT) * HAZARDLEN, m_ratesHazard.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputTimesHazard = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                               sizeof(TEST_DT) * HAZARDLEN, m_timesHazard.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputNominal = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * N,
                                           m_notional.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputRecovery = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * N,
                                            m_recovery.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputMaturity = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * N,
                                            m_maturity.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwInputFrequency = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                                             sizeof(TEST_DT) * N, m_frequency.data(), &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHwOutputBuffer = new cl::Buffer(*m_pContext, (CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE), sizeof(TEST_DT) * N,
                                           m_hostOutputBuffer.data(), &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int CreditDefaultSwap::releaseOCLObjects(void) {
    unsigned int i;

    if (m_pHwInputRatesIR != nullptr) {
        delete (m_pHwInputRatesIR);
        m_pHwInputRatesIR = nullptr;
    }

    if (m_pHwInputTimesIR != nullptr) {
        delete (m_pHwInputTimesIR);
        m_pHwInputTimesIR = nullptr;
    }

    if (m_pHwInputRatesHazard != nullptr) {
        delete (m_pHwInputRatesHazard);
        m_pHwInputRatesHazard = nullptr;
    }

    if (m_pHwInputTimesHazard != nullptr) {
        delete (m_pHwInputTimesHazard);
        m_pHwInputTimesHazard = nullptr;
    }

    if (m_pHwInputNominal != nullptr) {
        delete (m_pHwInputNominal);
        m_pHwInputNominal = nullptr;
    }

    if (m_pHwInputRecovery != nullptr) {
        delete (m_pHwInputRecovery);
        m_pHwInputRecovery = nullptr;
    }

    if (m_pHwInputMaturity != nullptr) {
        delete (m_pHwInputMaturity);
        m_pHwInputMaturity = nullptr;
    }

    if (m_pHwInputFrequency != nullptr) {
        delete (m_pHwInputFrequency);
        m_pHwInputFrequency = nullptr;
    }

    if (m_pHwOutputBuffer != nullptr) {
        delete (m_pHwOutputBuffer);
        m_pHwOutputBuffer = nullptr;
    }

    if (m_pCDSKernel != nullptr) {
        delete (m_pCDSKernel);
        m_pCDSKernel = nullptr;
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

int CreditDefaultSwap::run(float* timesIR,
                           float* ratesIR,
                           float* timesHazard,
                           float* ratesHazard,
                           float* notional,
                           float* recovery,
                           float* maturity,
                           int* frequency,
                           float* cdsSpread) {
    int retval = XLNX_OK;

    if (deviceIsPrepared()) {
        m_runStartTime = std::chrono::high_resolution_clock::now();

        // prepare the input data
        for (int i = 0; i < IRLEN; i++) {
            m_ratesIR[i] = ratesIR[i];
            m_timesIR[i] = timesIR[i];
        }

        for (int i = 0; i < HAZARDLEN; i++) {
            m_ratesHazard[i] = ratesHazard[i];
            m_timesHazard[i] = timesHazard[i];
        }

        for (int i = 0; i < N; i++) {
            m_notional[i] = notional[i];
            m_recovery[i] = recovery[i];
            m_maturity[i] = maturity[i];
            m_frequency[i] = frequency[i];
        }

        std::vector<cl::Memory> obIn0;
        obIn0.push_back(*m_pHwInputTimesIR);
        obIn0.push_back(*m_pHwInputRatesIR);
        obIn0.push_back(*m_pHwInputTimesHazard);
        obIn0.push_back(*m_pHwInputRatesHazard);
        obIn0.push_back(*m_pHwInputNominal);
        obIn0.push_back(*m_pHwInputRecovery);
        obIn0.push_back(*m_pHwInputMaturity);
        obIn0.push_back(*m_pHwInputFrequency);

        // output vector depedant on test case
        std::vector<cl::Memory> obOut0;
        obOut0.push_back(*m_pHwOutputBuffer);

        // Set the arguments
        m_pCDSKernel->setArg(0, *m_pHwInputTimesIR);
        m_pCDSKernel->setArg(1, *m_pHwInputRatesIR);
        m_pCDSKernel->setArg(2, *m_pHwInputTimesHazard);
        m_pCDSKernel->setArg(3, *m_pHwInputRatesHazard);
        m_pCDSKernel->setArg(4, *m_pHwInputNominal);
        m_pCDSKernel->setArg(5, *m_pHwInputRecovery);
        m_pCDSKernel->setArg(6, *m_pHwInputMaturity);
        m_pCDSKernel->setArg(7, *m_pHwInputFrequency);
        m_pCDSKernel->setArg(8, *m_pHwOutputBuffer);

        // Copy input data to device global memory
        m_pCommandQueue->enqueueMigrateMemObjects(obIn0, 0);
        m_pCommandQueue->finish();

        // Launch the Kernel
        m_pCommandQueue->enqueueTask(*m_pCDSKernel);
        m_pCommandQueue->finish();

        // Copy Result from Device Global Memory to Host Local Memory
        m_pCommandQueue->enqueueMigrateMemObjects(obOut0, CL_MIGRATE_MEM_OBJECT_HOST);
        m_pCommandQueue->finish();

        // --------------------------------
        // Give the caller back the results
        // --------------------------------
        for (int i = 0; i < N; i++) {
            cdsSpread[i] = m_hostOutputBuffer[i];
        }
        m_runEndTime = std::chrono::high_resolution_clock::now();

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int CreditDefaultSwap::getLastRunTime(void) {
    long long int duration = 0;
    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();
    return duration;
}
