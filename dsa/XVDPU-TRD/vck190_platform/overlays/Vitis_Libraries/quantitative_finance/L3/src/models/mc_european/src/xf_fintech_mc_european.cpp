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

#include "models/xf_fintech_mc_european.hpp"

#include "xf_fintech_mc_european_kernel_constants.hpp"

using namespace xf::fintech;

// const char* MCEuropean::KERNEL_NAMES[] = {"kernel_mc_0", "kernel_mc_1", "kernel_mc_2", "kernel_mc_3"};
const char* MCEuropean::KERNEL_NAMES[] = {"mc_euro_k"};

typedef struct _XCLBINLookupElement {
    Device::DeviceType deviceType;
    std::string xclbinName;
} XCLBINLookupElement;

static XCLBINLookupElement XCLBIN_LOOKUP_TABLE[] = {
    {Device::DeviceType::U50, "mc_euro_k.xclbin"}, // BUG-257 changing names to match output from L2
    {Device::DeviceType::U200, "mc_euro_k.xclbin"},
    {Device::DeviceType::U250, "mc_euro_k.xclbin"},
    {Device::DeviceType::U280, "mc_euro_k.xclbin"}};

static const unsigned int NUM_XCLBIN_LOOKUP_TABLE_ENTRIES =
    sizeof(XCLBIN_LOOKUP_TABLE) / sizeof(XCLBIN_LOOKUP_TABLE[0]);

MCEuropean::MCEuropean(std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;

    for (int i = 0; i < NUM_KERNELS; i++) {
        m_pKernels[i] = nullptr;
        m_hostOutputBuffers[i] = nullptr;
        m_pHWBuffers[i] = nullptr;
    }
    m_pSeedBuf = nullptr;

    m_xclbin_file = xclbin_file;
}

MCEuropean::~MCEuropean() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string MCEuropean::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int MCEuropean::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    unsigned int i;
    cl_int cl_retval = CL_SUCCESS;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    aligned_allocator<KDataType> allocator;
    aligned_allocator<unsigned int> allocator_seed;
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

    if (cl_retval == CL_SUCCESS) {
        m_hostSeed = allocator_seed.allocate(2);
        m_hostSeed[0] = 1;
        m_hostSeed[1] = 10001;
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
            m_hwSeed = {7, m_hostSeed, m_pKernel()};
            break;

        default:
        case Device::DeviceType::U250:
            for (i = 0; i < NUM_KERNELS; i++) {
                m_hwBufferOptions[i] = {8, m_hostOutputBuffers[i], m_pKernel()};
            }
            m_hwSeed = {7, m_hostSeed, m_pKernel()};
            break;
    }

    ///////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////

    if (cl_retval == CL_SUCCESS) {
        for (i = 0; i < NUM_KERNELS; i++) {
            m_pHWBuffers[i] =
                new cl::Buffer(*m_pContext, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                               (size_t)(OUTDEP * sizeof(KDataType)), &m_hwBufferOptions[i], &cl_retval);

            if (cl_retval != CL_SUCCESS) {
                break; // out of loop
            }
        }
    }

    if (cl_retval == CL_SUCCESS) {
        m_pSeedBuf = new cl::Buffer(*m_pContext, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                    sizeof(unsigned int), &m_hwSeed, &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int MCEuropean::releaseOCLObjects(void) {
    int retval = XLNX_OK;
    unsigned int i;
    aligned_allocator<KDataType> allocator;
    aligned_allocator<unsigned int> allocator_seed;

    for (i = 0; i < NUM_KERNELS; i++) {
        if (m_pHWBuffers[i] != nullptr) {
            delete (m_pHWBuffers[i]);
            m_pHWBuffers[i] = nullptr;
        }

        if (m_pSeedBuf != nullptr) {
            delete (m_pSeedBuf);
            m_pSeedBuf = nullptr;
        }

        if (m_hostOutputBuffers[i] != nullptr) {
            allocator.deallocate((KDataType*)(m_hostOutputBuffers[i]), OUTDEP);
            m_hostOutputBuffers[i] = nullptr;
        }

        if (m_hostSeed != nullptr) {
            allocator_seed.deallocate((unsigned int*)(m_hostSeed), 2);
            m_hostSeed = nullptr;
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

// SINGLE asset, run to TOLERANCE
int MCEuropean::run(OptionType optionType,
                    double stockPrice,
                    double strikePrice,
                    double riskFreeRate,
                    double dividendYield,
                    double volatility,
                    double timeToMaturity,
                    double requiredTolerance,
                    double* pOptionPrice) {
    int retval = XLNX_OK;
    unsigned int requiredSamples;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

    // since this method only exposes requiredTolerance, we must set
    // requiredSamples = 0
    requiredSamples = 0;

    retval = runInternal(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility, timeToMaturity,
                         requiredTolerance, requiredSamples, pOptionPrice);

    return retval;
}

// SINGLE asset, run to REQUIRED NUM SAMPLES
int MCEuropean::run(OptionType optionType,
                    double stockPrice,
                    double strikePrice,
                    double riskFreeRate,
                    double dividendYield,
                    double volatility,
                    double timeToMaturity,
                    unsigned int requiredSamples,
                    double* pOptionPrice) {
    int retval = XLNX_OK;
    double requiredTolerance;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

    // since this method only exposes requiredSamples, we must set
    // requiredTolerance = 0.0
    requiredTolerance = 0.0;

    retval = runInternal(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield, volatility, timeToMaturity,
                         requiredTolerance, requiredSamples, pOptionPrice);
    return retval;
}

// MULTI asset, run to TOLERANCE
int MCEuropean::run(OptionType* optionType,
                    double* stockPrice,
                    double* strikePrice,
                    double* riskFreeRate,
                    double* dividendYield,
                    double* volatility,
                    double* timeToMaturity,
                    double* requiredTolerance,
                    double* outputOptionPrice,
                    unsigned int numAssets) {
    int retval = XLNX_OK;
    unsigned int requiredSamples[NUM_KERNELS] = {0};
    unsigned int numKernelsToRun;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

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
                             &outputOptionPrice[i], numKernelsToRun);
    }

    return retval;
}

// MULTI asset, run to REQUIRED NUM SAMPLES
int MCEuropean::run(OptionType* optionType,
                    double* stockPrice,
                    double* strikePrice,
                    double* riskFreeRate,
                    double* dividendYield,
                    double* volatility,
                    double* timeToMaturity,
                    unsigned int* requiredSamples,
                    double* outputOptionPrice,
                    unsigned int numAssets) {
    int retval = XLNX_OK;
    double requiredTolerance[NUM_KERNELS] = {0.0};
    unsigned int numKernelsToRun;

    // The kernels take in BOTH requiredTolerance AND requiredSamples.
    // However only ONE is used during processing...
    // If requiredSamples > 0, the model will run for that number of samples
    // If requiredSamples == 0, the model will run for as long as necessary to
    // meet requiredTolerance

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
                             &requiredSamples[i], &outputOptionPrice[i], numKernelsToRun);
    }

    return retval;
}

int MCEuropean::runInternal(OptionType optionType,
                            double stockPrice,
                            double strikePrice,
                            double riskFreeRate,
                            double dividendYield,
                            double volatility,
                            double timeToMaturity,
                            double requiredTolerance,
                            unsigned int requiredSamples,
                            double* pOptionPrice) {
    int retval = XLNX_OK;
    KDataType totalOutput = 0.0;
    unsigned int i;

    OptionType multipleOptionType[NUM_KERNELS];
    double multipleStockPrice[NUM_KERNELS];
    double multipleStrikePrice[NUM_KERNELS];
    double multipleRiskFreeRate[NUM_KERNELS];
    double multipleDividendYield[NUM_KERNELS];
    double multipleVolatility[NUM_KERNELS];
    double multipleTimeToMaturity[NUM_KERNELS];
    double multipleRequiredTolerance[NUM_KERNELS];
    unsigned int multipleRequiredSamples[NUM_KERNELS];
    double multipleOptionPrice[NUM_KERNELS];

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        // Here we are setting up the arrays that represent the data we will pass to
        // the kernels.
        // The i'th element of each array will be passed to the i'th kernel.
        //
        // Since we are running with a SINGLE ASSET DATA here, we pass the SAME DATA
        // TO ALL KERNELS,
        // then at the end, we sum the output and take an average.

        for (i = 0; i < NUM_KERNELS; i++) {
            multipleOptionType[i] = optionType;
            multipleStockPrice[i] = (KDataType)stockPrice;
            multipleStrikePrice[i] = (KDataType)strikePrice;
            multipleRiskFreeRate[i] = (KDataType)riskFreeRate;
            multipleDividendYield[i] = (KDataType)dividendYield;
            multipleVolatility[i] = (KDataType)volatility;
            multipleTimeToMaturity[i] = (KDataType)timeToMaturity;
            multipleRequiredTolerance[i] = (KDataType)requiredTolerance;
            multipleRequiredSamples[i] = requiredSamples;
        }

        retval = runInternal(multipleOptionType, multipleStockPrice, multipleStrikePrice, multipleRiskFreeRate,
                             multipleDividendYield, multipleVolatility, multipleTimeToMaturity,
                             multipleRequiredTolerance, multipleRequiredSamples, multipleOptionPrice, NUM_KERNELS);

        // ---------------
        // Post-Processing
        // ---------------

        if (retval == XLNX_OK) {
            totalOutput = 0.0;

            // sum the option price output from each kernel...
            for (i = 0; i < NUM_KERNELS; i++) {
                totalOutput += multipleOptionPrice[i];
            }

            //...and take the average....
            *pOptionPrice = (double)(totalOutput / (KDataType)NUM_KERNELS);
        }

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

int MCEuropean::runInternal(OptionType* optionType,
                            double* stockPrice,
                            double* strikePrice,
                            double* riskFreeRate,
                            double* dividendYield,
                            double* volatility,
                            double* timeToMaturity,
                            double* requiredTolerance,
                            unsigned int* requiredSamples,
                            double* outputOptionPrice,
                            unsigned int numAssets) {
    int retval = XLNX_OK;
    unsigned int timeSteps = 1;
    unsigned int loop_nm = 1;
    KDataType totalOutput = 0.0;
    unsigned int i, j;
    std::vector<cl::Memory> outVector;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        for (i = 0; i < numAssets; i++) {
            m_pKernels[i]->setArg(0, (KDataType)stockPrice[i]);
            m_pKernels[i]->setArg(1, (KDataType)volatility[i]);
            m_pKernels[i]->setArg(2, (KDataType)dividendYield[i]);
            m_pKernels[i]->setArg(3, (KDataType)riskFreeRate[i]);
            m_pKernels[i]->setArg(4, (KDataType)timeToMaturity[i]);
            m_pKernels[i]->setArg(5, (KDataType)strikePrice[i]);
            m_pKernels[i]->setArg(6, (unsigned int)optionType[i]);
            m_pKernels[i]->setArg(7, *m_pSeedBuf);
            m_pKernels[i]->setArg(8, *m_pHWBuffers[i]);
            m_pKernels[i]->setArg(9, (KDataType)requiredTolerance[i]);
            m_pKernels[i]->setArg(10, requiredSamples[i]);
            m_pKernels[i]->setArg(11, timeSteps);

            m_pCommandQueue->enqueueTask(*(m_pKernels[i]), nullptr, nullptr);
        }

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        for (i = 0; i < numAssets; i++) {
            outVector.push_back(*(m_pHWBuffers[i]));
        }

        m_pCommandQueue->enqueueMigrateMemObjects(outVector, CL_MIGRATE_MEM_OBJECT_HOST, nullptr /*&kernel_events[i]*/,
                                                  nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        // ---------------
        // Post-Processing
        // ---------------

        for (i = 0; i < numAssets; i++) {
            KDataType* pBuffer = (KDataType*)(m_hostOutputBuffers[i]);

            totalOutput = (KDataType)0.0;

            // sum the outputs...
            for (j = 0; j < loop_nm; j++) {
                totalOutput += pBuffer[j];
            }

            outputOptionPrice[i] = (double)(totalOutput / (KDataType)loop_nm);
        }

    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

long long int MCEuropean::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();

    return duration;
}
