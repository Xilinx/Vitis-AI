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

#include "models/xf_fintech_cf_b76.hpp"

using namespace xf::fintech;

// For efficient memory bandwidth utilization, the HW kernel operates on data
// elements
// that are KERNEL_PARAMETER_WIDTH (512 at time of writing) bits wide.
//
// This is used for both input and output buffers.
//
// This means that each call to the kernel needs to pass pointers to buffers
// that at least 512 bits wide.
// i.e. we need our buffers to be allocated in chunks of KERNEL_PARAMETER_WIDTH
// / sizeof(float) elements.

const unsigned int CFB76::NUM_ELEMENTS_PER_BUFFER_CHUNK = CFB76::KERNEL_PARAMETER_BITWIDTH / sizeof(CFB76::KDataType);

CFB76::CFB76(unsigned int maxNumAssets, std::string xclbin_file) {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;
    m_pKernel = nullptr;
    m_xclbin_file = xclbin_file;

    this->allocateBuffers(maxNumAssets);
}

CFB76::~CFB76() {
    this->deallocateBuffers();

    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string CFB76::getXCLBINName(Device* device) {
    return m_xclbin_file;
}

int CFB76::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::string xclbinName;

    cl::Device clDevice;

    clDevice = device->getCLDevice();

    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    ///////////////////////////////
    // Create COMMAND QUEUE Object
    ///////////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(*m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &cl_retval);
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
    if (cl_retval == CL_SUCCESS) {
        m_pKernel = new cl::Kernel(*m_pProgram, "b76_kernel", &cl_retval);
    }

    /////////////////////////
    // Create BUFFER Objects
    /////////////////////////

    if (cl_retval == CL_SUCCESS) {
        m_pForwardPriceHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->forwardPrice, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pStrikePriceHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->strikePrice, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pVolatilityHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->volatility, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pRiskFreeRateHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->riskFreeRate, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pTimeToMaturityHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->timeToMaturity, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pOptionPriceHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->optionPrice, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pDeltaHWBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                          m_numPaddedBufferElements * sizeof(KDataType), this->delta, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pGammaHWBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                          m_numPaddedBufferElements * sizeof(KDataType), this->gamma, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pVegaHWBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                         m_numPaddedBufferElements * sizeof(KDataType), this->vega, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pThetaHWBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                          m_numPaddedBufferElements * sizeof(KDataType), this->theta, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pRhoHWBuffer = new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                        m_numPaddedBufferElements * sizeof(KDataType), this->rho, &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printCLError(cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int CFB76::releaseOCLObjects(void) {
    int retval = XLNX_OK;
    unsigned int i;

    if (m_pForwardPriceHWBuffer != nullptr) {
        delete (m_pForwardPriceHWBuffer);
        m_pForwardPriceHWBuffer = nullptr;
    }

    if (m_pStrikePriceHWBuffer != nullptr) {
        delete (m_pStrikePriceHWBuffer);
        m_pStrikePriceHWBuffer = nullptr;
    }

    if (m_pVolatilityHWBuffer != nullptr) {
        delete (m_pVolatilityHWBuffer);
        m_pVolatilityHWBuffer = nullptr;
    }

    if (m_pRiskFreeRateHWBuffer != nullptr) {
        delete (m_pRiskFreeRateHWBuffer);
        m_pRiskFreeRateHWBuffer = nullptr;
    }

    if (m_pTimeToMaturityHWBuffer != nullptr) {
        delete (m_pTimeToMaturityHWBuffer);
        m_pTimeToMaturityHWBuffer = nullptr;
    }

    if (m_pOptionPriceHWBuffer != nullptr) {
        delete (m_pOptionPriceHWBuffer);
        m_pOptionPriceHWBuffer = nullptr;
    }

    if (m_pDeltaHWBuffer != nullptr) {
        delete (m_pDeltaHWBuffer);
        m_pDeltaHWBuffer = nullptr;
    }

    if (m_pGammaHWBuffer != nullptr) {
        delete (m_pGammaHWBuffer);
        m_pGammaHWBuffer = nullptr;
    }

    if (m_pVegaHWBuffer != nullptr) {
        delete (m_pVegaHWBuffer);
        m_pVegaHWBuffer = nullptr;
    }

    if (m_pThetaHWBuffer != nullptr) {
        delete (m_pThetaHWBuffer);
        m_pThetaHWBuffer = nullptr;
    }

    if (m_pRhoHWBuffer != nullptr) {
        delete (m_pRhoHWBuffer);
        m_pRhoHWBuffer = nullptr;
    }

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

    return retval;
}

void CFB76::allocateBuffers(unsigned int numRequestedElements) {
    aligned_allocator<KDataType> allocator;

    m_numPaddedBufferElements = calculatePaddedNumElements(numRequestedElements);

    this->forwardPrice = allocator.allocate(m_numPaddedBufferElements);
    this->strikePrice = allocator.allocate(m_numPaddedBufferElements);
    this->volatility = allocator.allocate(m_numPaddedBufferElements);
    this->riskFreeRate = allocator.allocate(m_numPaddedBufferElements);
    this->timeToMaturity = allocator.allocate(m_numPaddedBufferElements);

    this->optionPrice = allocator.allocate(m_numPaddedBufferElements);

    this->delta = allocator.allocate(m_numPaddedBufferElements);
    this->gamma = allocator.allocate(m_numPaddedBufferElements);
    this->vega = allocator.allocate(m_numPaddedBufferElements);
    this->theta = allocator.allocate(m_numPaddedBufferElements);
    this->rho = allocator.allocate(m_numPaddedBufferElements);
}

void CFB76::deallocateBuffers(void) {
    aligned_allocator<KDataType> allocator;

    if (this->forwardPrice != nullptr) {
        allocator.deallocate(this->forwardPrice, m_numPaddedBufferElements);
        this->forwardPrice = nullptr;
    }

    if (this->strikePrice != nullptr) {
        allocator.deallocate(this->strikePrice, m_numPaddedBufferElements);
        this->strikePrice = nullptr;
    }

    if (this->volatility != nullptr) {
        allocator.deallocate(this->volatility, m_numPaddedBufferElements);
        this->volatility = nullptr;
    }

    if (this->riskFreeRate != nullptr) {
        allocator.deallocate(this->riskFreeRate, m_numPaddedBufferElements);
        this->riskFreeRate = nullptr;
    }

    if (this->timeToMaturity != nullptr) {
        allocator.deallocate(this->timeToMaturity, m_numPaddedBufferElements);
        this->timeToMaturity = nullptr;
    }

    if (this->optionPrice != nullptr) {
        allocator.deallocate(this->optionPrice, m_numPaddedBufferElements);
        this->optionPrice = nullptr;
    }

    if (this->delta != nullptr) {
        allocator.deallocate(this->delta, m_numPaddedBufferElements);
        this->delta = nullptr;
    }

    if (this->gamma != nullptr) {
        allocator.deallocate(this->gamma, m_numPaddedBufferElements);
        this->gamma = nullptr;
    }

    if (this->vega != nullptr) {
        allocator.deallocate(this->vega, m_numPaddedBufferElements);
        this->vega = nullptr;
    }

    if (this->theta != nullptr) {
        allocator.deallocate(this->theta, m_numPaddedBufferElements);
        this->theta = nullptr;
    }

    if (this->rho != nullptr) {
        allocator.deallocate(this->rho, m_numPaddedBufferElements);
        this->rho = nullptr;
    }

    m_numPaddedBufferElements = 0;
}

unsigned int CFB76::calculatePaddedNumElements(unsigned int numRequestedElements) {
    unsigned int numChunks;
    unsigned int numPaddedElements;

    // due to the way the HW processes data, the number of elements in a buffer
    // needs to be multiples of NUM_ELEMENTS_PER_BUFFER_CHUNK.
    // so we need to round up the amount to the next nearest whole number of
    // chunks

    numChunks = numRequestedElements + (NUM_ELEMENTS_PER_BUFFER_CHUNK - 1) / NUM_ELEMENTS_PER_BUFFER_CHUNK;

    numPaddedElements = numChunks * NUM_ELEMENTS_PER_BUFFER_CHUNK;

    return numPaddedElements;
}

int CFB76::run(OptionType optionType, unsigned int numAssets) {
    int retval = XLNX_OK;
    unsigned int optionFlag;
    std::vector<cl::Memory> inputVector;
    std::vector<cl::Memory> outputVector;

    unsigned int numPaddedAssets;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (optionType == OptionType::Call) {
        optionFlag = 1;
    } else {
        optionFlag = 0;
    }

    numPaddedAssets = calculatePaddedNumElements(numAssets);

    m_pKernel->setArg(0, (*m_pForwardPriceHWBuffer));
    m_pKernel->setArg(1, (*m_pVolatilityHWBuffer));
    m_pKernel->setArg(2, (*m_pRiskFreeRateHWBuffer));
    m_pKernel->setArg(3, (*m_pTimeToMaturityHWBuffer));
    m_pKernel->setArg(4, (*m_pStrikePriceHWBuffer));
    m_pKernel->setArg(5, optionFlag);
    m_pKernel->setArg(6, numPaddedAssets);
    m_pKernel->setArg(7, (*m_pOptionPriceHWBuffer));
    m_pKernel->setArg(8, (*m_pDeltaHWBuffer));
    m_pKernel->setArg(9, (*m_pGammaHWBuffer));
    m_pKernel->setArg(10, (*m_pVegaHWBuffer));
    m_pKernel->setArg(11, (*m_pThetaHWBuffer));
    m_pKernel->setArg(12, (*m_pRhoHWBuffer));

    inputVector.push_back((*m_pForwardPriceHWBuffer));
    inputVector.push_back((*m_pVolatilityHWBuffer));
    inputVector.push_back((*m_pRiskFreeRateHWBuffer));
    inputVector.push_back((*m_pTimeToMaturityHWBuffer));
    inputVector.push_back((*m_pStrikePriceHWBuffer));

    m_pCommandQueue->enqueueMigrateMemObjects(inputVector, 0, nullptr /*&kernel_events[i]*/, nullptr);

    m_pCommandQueue->enqueueTask(*m_pKernel);

    m_pCommandQueue->flush();
    m_pCommandQueue->finish();

    outputVector.push_back((*m_pOptionPriceHWBuffer));
    outputVector.push_back((*m_pDeltaHWBuffer));
    outputVector.push_back((*m_pGammaHWBuffer));
    outputVector.push_back((*m_pVegaHWBuffer));
    outputVector.push_back((*m_pThetaHWBuffer));
    outputVector.push_back((*m_pRhoHWBuffer));

    m_pCommandQueue->enqueueMigrateMemObjects(outputVector, CL_MIGRATE_MEM_OBJECT_HOST, nullptr /*&kernel_events[i]*/,
                                              nullptr);

    m_pCommandQueue->flush();
    m_pCommandQueue->finish();

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

long long int CFB76::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();

    return duration;
}
