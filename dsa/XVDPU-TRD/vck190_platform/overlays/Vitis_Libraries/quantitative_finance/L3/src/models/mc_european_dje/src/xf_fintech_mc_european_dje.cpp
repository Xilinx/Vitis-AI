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

#include <limits.h>

#include "xf_fintech_error_codes.hpp"
#include "xf_fintech_trace.hpp"

#include "models/xf_fintech_mc_european_dje.hpp"

#include "xf_fintech_mc_european_dje_kernel_constants.hpp"

using namespace xf::fintech;

const char* MCEuropeanDJE::KERNEL_NAMES[] = {"kernel_mc_0"};

typedef struct _XCLBINLookupElement {
    Device::DeviceType deviceType;
    std::string xclbinName;
} XCLBINLookupElement;

static XCLBINLookupElement XCLBIN_LOOKUP_TABLE[] = {{Device::DeviceType::U50, "mc_european_dowjones.xclbin"},
                                                    {Device::DeviceType::U200, "mc_european_dowjones.xclbin"},
                                                    {Device::DeviceType::U250, "mc_european_dowjones.xclbin"},
                                                    {Device::DeviceType::U280, "mc_european_dowjones.xclbin"}};

static const unsigned int NUM_XCLBIN_LOOKUP_TABLE_ENTRIES =
    sizeof(XCLBIN_LOOKUP_TABLE) / sizeof(XCLBIN_LOOKUP_TABLE[0]);

MCEuropeanDJE::MCEuropeanDJE(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;

    for (int i = 0; i < NUM_KERNELS; i++) {
        m_pKernels[i] = nullptr;
        m_hostOutputBuffers[i] = nullptr;
        m_pHWBuffers[i] = nullptr;
    }

    m_xclbin_file = xclbin_file;
}

MCEuropeanDJE::~MCEuropeanDJE() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string MCEuropeanDJE::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int MCEuropeanDJE::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    unsigned int i;
    cl_int cl_retval = CL_SUCCESS;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    aligned_allocator<KDataType> allocator;
    std::string xclbinName;

    cl::Device clDevice;

    clDevice = device->getCLDevice();

    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    ///////////////////////////////
    // Create COMMAND QUEUE Object
    ///////////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(
            *m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_retval);
    }

    /////////////////
    // Import XCLBIN
    /////////////////
    if (cl_retval == CL_SUCCESS) {
        start = std::chrono::high_resolution_clock::now();

        xclbinName = getXCLBINName(device);

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
    cl::Kernel m_pKernel(*m_pProgram, KERNEL_NAMES[0], &cl_retval);
    if (cl_retval == CL_SUCCESS) {
        for (i = 0; i < NUM_KERNELS; i++) {
            m_pKernels[i] = new cl::Kernel(*m_pProgram, KERNEL_NAMES[i], &cl_retval);

            if (cl_retval != CL_SUCCESS) {
                break; // out of loop
            }
        }
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        for (i = 0; i < NUM_KERNELS; i++) {
            m_hostOutputBuffers[i] = allocator.allocate(OUTDEP);

            if (m_hostOutputBuffers[i] == nullptr) {
                cl_retval = CL_OUT_OF_HOST_MEMORY;
                break; // out of loop
            }
        }
    }

    ////////////////////////////
    // Setup HW BUFFER OPTIONS
    ////////////////////////////
    Device::DeviceType deviceType = device->getDeviceType();

    switch (deviceType) {
        case Device::DeviceType::U200:
            for (i = 0; i < NUM_KERNELS; i++) {
                m_hwBufferOptions[i] = {8, m_hostOutputBuffers[i], m_pKernel()};
            }
            break;

        default:
        case Device::DeviceType::U250:
            for (i = 0; i < NUM_KERNELS; i++) {
                m_hwBufferOptions[i] = {8, m_hostOutputBuffers[i], m_pKernel()};
            }
            break;
    }

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////
    if (cl_retval == CL_SUCCESS) {
        for (i = 0; i < NUM_KERNELS; i++) {
            m_pHWBuffers[i] =
                new cl::Buffer(*m_pContext, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(OUTDEP * sizeof(KDataType)), &m_hwBufferOptions[i], &cl_retval);

            if (cl_retval != CL_SUCCESS) {
                break; // out of loop
            }
        }
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int MCEuropeanDJE::releaseOCLObjects(void) {
    int retval = XLNX_OK;
    unsigned int i;
    aligned_allocator<KDataType> allocator;

    for (i = 0; i < NUM_KERNELS; i++) {
        if (m_pHWBuffers[i] != nullptr) {
            delete (m_pHWBuffers[i]);
            m_pHWBuffers[i] = nullptr;
        }

        if (m_hostOutputBuffers[i] != nullptr) {
            allocator.deallocate((KDataType*)(m_hostOutputBuffers[i]), OUTDEP);
            m_hostOutputBuffers[i] = nullptr;
        }

        if (m_pKernels[i] != nullptr) {
            delete (m_pKernels[i]);
            m_pKernels[i] = nullptr;
        }
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

    return retval;
}

// MULTI asset, run to TOLERANCE
int MCEuropeanDJE::run(OptionType* optionType,
                       double* stockPrice,
                       double* strikePrice,
                       double* riskFreeRate,
                       double* dividendYield,
                       double* volatility,
                       double* timeToMaturity,
                       double* requiredTolerance,
                       unsigned int numAssets,
                       double dowDivisor,
                       double* DJIAOutput) {
    int retval = XLNX_OK;
    unsigned int requiredSamples[NUM_KERNELS] = {0};

    unsigned int numKernelsToRun;
    double optionValues[NUM_KERNELS];
    double totalOptionsValue = 0.0;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

    m_runStartTime = std::chrono::high_resolution_clock::now();

    // We will process the asset data in NUM_KERNELS sized chunks...
    for (unsigned int i = 0; i < numAssets; i += NUM_KERNELS) {
        numKernelsToRun = numAssets - i;

        // need to limit each run to the number of kernels that are present...
        if (numKernelsToRun > NUM_KERNELS) {
            numKernelsToRun = NUM_KERNELS;
        }

        retval = runInternal(&optionType[i], &stockPrice[i], &strikePrice[i], &riskFreeRate[i], &dividendYield[i],
                             &volatility[i], &timeToMaturity[i], &requiredTolerance[i],
                             requiredSamples, // <-- remember this is a LOCAL variable
                             optionValues, numKernelsToRun);

        if (retval == XLNX_OK) {
            // Need to keep a total of all the returned option values...
            for (unsigned int j = 0; j < numKernelsToRun; j++) {
                totalOptionsValue += optionValues[j];
            }
        }
    }

    if (retval == XLNX_OK) {
        *DJIAOutput = totalOptionsValue / dowDivisor / 100.0; // TODO do we need this extra division by 100.0?
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

// MULTI asset, run to REQUIRED NUM SAMPLES
int MCEuropeanDJE::run(OptionType* optionType,
                       double* stockPrice,
                       double* strikePrice,
                       double* riskFreeRate,
                       double* dividendYield,
                       double* volatility,
                       double* timeToMaturity,
                       unsigned int* requiredSamples,
                       unsigned int numAssets,
                       double dowDivisor,
                       double* DJIAOutput) {
    int retval = XLNX_OK;
    double requiredTolerance[NUM_KERNELS] = {0.0};
    unsigned int numKernelsToRun;
    double optionValues[NUM_KERNELS];
    double totalOptionsValue = 0.0;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

    m_runStartTime = std::chrono::high_resolution_clock::now();

    // We will process the asset data in NUM_KERNELS sized chunks...
    for (unsigned int i = 0; i < numAssets; i += NUM_KERNELS) {
        numKernelsToRun = numAssets - i;

        // need to limit each run to the number of kernels that are present...
        if (numKernelsToRun > NUM_KERNELS) {
            numKernelsToRun = NUM_KERNELS;
        }

        retval = runInternal(&optionType[i], &stockPrice[i], &strikePrice[i], &riskFreeRate[i], &dividendYield[i],
                             &volatility[i], &timeToMaturity[i],
                             requiredTolerance, // <-- remember this is a LOCAL variable
                             &requiredSamples[i], optionValues, numKernelsToRun);

        if (retval == XLNX_OK) {
            // Need to keep a total of all the returned option values...
            for (unsigned int j = 0; j < numKernelsToRun; j++) {
                totalOptionsValue += optionValues[j];
            }
        } else {
            break; // out of loop
        }
    }

    if (retval == XLNX_OK) {
        *DJIAOutput = totalOptionsValue / dowDivisor / 100.0; // TODO do we need this extra division by 100.0?
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

int MCEuropeanDJE::runInternal(OptionType* optionType,
                               double* stockPrice,
                               double* strikePrice,
                               double* riskFreeRate,
                               double* dividendYield,
                               double* volatility,
                               double* timeToMaturity,
                               double* requiredTolerance,
                               unsigned int* requiredSamples,
                               double* outputOptionValues,
                               unsigned int numAssets) {
    int retval = XLNX_OK;
    unsigned int timeSteps = 1;
    unsigned int loop_nm = 1;
    unsigned int maxSamples = 0;
    KDataType totalOutput = 0.0;
    unsigned int i, j;

    if (deviceIsPrepared()) {
        for (i = 0; i < numAssets; i++) {
            m_pKernels[i]->setArg(0, loop_nm);
            m_pKernels[i]->setArg(1, (KDataType)stockPrice[i]);
            m_pKernels[i]->setArg(2, (KDataType)volatility[i]);
            m_pKernels[i]->setArg(3, (KDataType)dividendYield[i]);
            m_pKernels[i]->setArg(4, (KDataType)riskFreeRate[i]);
            m_pKernels[i]->setArg(5, (KDataType)timeToMaturity[i]);
            m_pKernels[i]->setArg(6, (KDataType)strikePrice[i]);
            m_pKernels[i]->setArg(7, (bool)optionType[i]);
            m_pKernels[i]->setArg(8, *(m_pHWBuffers[i]));
            m_pKernels[i]->setArg(9, (KDataType)requiredTolerance[i]);
            m_pKernels[i]->setArg(10, requiredSamples[i]);
            m_pKernels[i]->setArg(11, timeSteps);
            m_pKernels[i]->setArg(12, maxSamples);

            m_pCommandQueue->enqueueTask(*(m_pKernels[i]), nullptr, nullptr);
        }

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        std::vector<cl::Memory> outVector;

        for (i = 0; i < numAssets; i++) {
            outVector.push_back(*(m_pHWBuffers[i]));
        }

        m_pCommandQueue->enqueueMigrateMemObjects(outVector, CL_MIGRATE_MEM_OBJECT_HOST, nullptr /*&kernel_events[i]*/,
                                                  nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        for (i = 0; i < numAssets; i++) {
            KDataType* pBuffer = (KDataType*)(m_hostOutputBuffers[i]);

            totalOutput = (KDataType)0.0;

            // sum the outputs...
            for (j = 0; j < loop_nm; j++) {
                totalOutput += pBuffer[j];
            }

            outputOptionValues[i] = (double)(totalOutput / (KDataType)loop_nm);
        }

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    return retval;
}

long long int MCEuropeanDJE::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();

    return duration;
}
