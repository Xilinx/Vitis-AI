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

#include "models/xf_fintech_cf_garman_kohlhagen.hpp"

using namespace xf::fintech;

typedef struct _XCLBINLookupElement {
    Device::DeviceType deviceType;
    std::string xclbinName;
} XCLBINLookupElement;

static XCLBINLookupElement XCLBIN_LOOKUP_TABLE[] = {{Device::DeviceType::U50, "gk_kernel.xclbin"},
                                                    {Device::DeviceType::U200, "gk_kernel.xclbin"},
                                                    {Device::DeviceType::U250, "gk_kernel.xclbin"},
                                                    {Device::DeviceType::U280, "gk_kernel.xclbin"}};

static const unsigned int NUM_XCLBIN_LOOKUP_TABLE_ENTRIES =
    sizeof(XCLBIN_LOOKUP_TABLE) / sizeof(XCLBIN_LOOKUP_TABLE[0]);

CFGarmanKohlhagen::CFGarmanKohlhagen(unsigned int maxNumAssets, std::string xclbin_file)
    : CFBlackScholes(maxNumAssets, xclbin_file) {
    allocateBuffers(maxNumAssets);
}

CFGarmanKohlhagen::~CFGarmanKohlhagen() {
    aligned_allocator<KDataType> allocator;

    if (this->domesticRate != nullptr) {
        allocator.deallocate(this->domesticRate, m_numPaddedBufferElements);
        this->domesticRate = nullptr;
    }

    if (this->foreignRate != nullptr) {
        allocator.deallocate(this->foreignRate, m_numPaddedBufferElements);
        this->foreignRate = nullptr;
    }
}

int CFGarmanKohlhagen::releaseOCLObjects(void) {
    int retval = XLNX_OK;

    retval = CFBlackScholes::releaseOCLObjects();

    if (m_pDomesticRateHWBuffer != nullptr) {
        delete (m_pDomesticRateHWBuffer);
        m_pDomesticRateHWBuffer = nullptr;
    }

    if (m_pForeignRateHWBuffer != nullptr) {
        delete (m_pForeignRateHWBuffer);
        m_pForeignRateHWBuffer = nullptr;
    }

    return retval;
}

int CFGarmanKohlhagen::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;

    retval = CFBlackScholes::createOCLObjects(device);

    if (retval == XLNX_OK) {
        m_pDomesticRateHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->domesticRate, &cl_retval);
    }

    if (retval == XLNX_OK && cl_retval == CL_SUCCESS) {
        m_pForeignRateHWBuffer =
            new cl::Buffer(*m_pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           m_numPaddedBufferElements * sizeof(KDataType), this->foreignRate, &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printCLError(cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

void CFGarmanKohlhagen::allocateBuffers(unsigned int numRequestedElements) {
    aligned_allocator<KDataType> allocator;

    this->domesticRate = allocator.allocate(m_numPaddedBufferElements);
    this->foreignRate = allocator.allocate(m_numPaddedBufferElements);
}

int CFGarmanKohlhagen::run(OptionType optionType, unsigned int numAssets) {
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

    m_pKernel->setArg(0, (*m_pStockPriceHWBuffer));
    m_pKernel->setArg(1, (*m_pVolatilityHWBuffer));
    m_pKernel->setArg(2, (*m_pDomesticRateHWBuffer));
    m_pKernel->setArg(3, (*m_pTimeToMaturityHWBuffer));
    m_pKernel->setArg(4, (*m_pStrikePriceHWBuffer));
    m_pKernel->setArg(5, (*m_pForeignRateHWBuffer));
    m_pKernel->setArg(6, optionFlag);
    m_pKernel->setArg(7, numPaddedAssets);
    m_pKernel->setArg(8, (*m_pOptionPriceHWBuffer));
    m_pKernel->setArg(9, (*m_pDeltaHWBuffer));
    m_pKernel->setArg(10, (*m_pGammaHWBuffer));
    m_pKernel->setArg(11, (*m_pVegaHWBuffer));
    m_pKernel->setArg(12, (*m_pThetaHWBuffer));
    m_pKernel->setArg(13, (*m_pRhoHWBuffer));

    inputVector.push_back((*m_pStockPriceHWBuffer));
    inputVector.push_back((*m_pVolatilityHWBuffer));
    inputVector.push_back((*m_pDomesticRateHWBuffer));
    inputVector.push_back((*m_pTimeToMaturityHWBuffer));
    inputVector.push_back((*m_pStrikePriceHWBuffer));
    inputVector.push_back((*m_pForeignRateHWBuffer));

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

static const char* gkKernelName = "gk_kernel";
const char* CFGarmanKohlhagen::getKernelName() {
    return gkKernelName;
}
