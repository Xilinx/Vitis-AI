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

#include "models/xf_fintech_mc_american.hpp"

#include "xf_fintech_mc_american_kernel_constants.hpp"

using namespace xf::fintech;

static const size_t PRICE_ELEMENT_SIZE = sizeof(KDataType) * UN_K1;
static const size_t MATRIX_ELEMENT_SIZE = sizeof(KDataType);
static const size_t COEFF_ELEMENT_SIZE = sizeof(KDataType) * COEF;

static const size_t PRICE_NUM_ELEMENTS = DEPTH_P;
static const size_t MATRIX_NUM_ELEMENTS = DEPTH_M;
static const size_t COEFF_NUM_ELEMENTS = TIMESTEPS - 1;

typedef struct _XCLBINLookupElement {
    Device::DeviceType deviceType;
    std::string xclbinName;
} XCLBINLookupElement;

// NOTE - we are using the multi-kernel version of the MCAmerican engine...
static XCLBINLookupElement XCLBIN_LOOKUP_TABLE[] = {
    {Device::DeviceType::U50, "MCAE_k.xclbin"}, // BUG-257 changing names to match output from L2
    {Device::DeviceType::U200, "MCAE_k.xclbin"},
    {Device::DeviceType::U250, "MCAE_k.xclbin"},
    {Device::DeviceType::U280, "MCAE_k.xclbin"}};

static const unsigned int NUM_XCLBIN_LOOKUP_TABLE_ENTRIES =
    sizeof(XCLBIN_LOOKUP_TABLE) / sizeof(XCLBIN_LOOKUP_TABLE[0]);

MCAmerican::MCAmerican() {
    m_pContext = nullptr;
    m_pCommandQueue = nullptr;
    m_pProgram = nullptr;

    m_hostOutputPricesBuffer = nullptr;
    m_hostOutputMatrixBuffer = nullptr;
    m_hostCoeffBuffer = nullptr;
    m_hostOutputBuffer1 = nullptr;
    m_hostOutputBuffer2 = nullptr;

    m_pPreSampleKernel = nullptr;
    m_pCalibrationKernel = nullptr;
    m_pPricingKernel1 = nullptr;
    m_pPricingKernel2 = nullptr;

    m_pHWOutputPriceBuffer = nullptr;
    m_pHWOutputMatrixBuffer = nullptr;
    m_pHWCoeffBuffer = nullptr;
    m_pHWOutputBuffer1 = nullptr;
    m_pHWOutputBuffer2 = nullptr;
}

MCAmerican::~MCAmerican() {
    if (deviceIsPrepared()) {
        releaseDevice();
    }
}

std::string MCAmerican::getXCLBINName(Device* device) {
    std::string xclbinName = "UNSUPPORTED_DEVICE";
    Device::DeviceType deviceType;
    unsigned int i;
    XCLBINLookupElement* pElement;

    deviceType = device->getDeviceType();

    for (i = 0; i < NUM_XCLBIN_LOOKUP_TABLE_ENTRIES; i++) {
        pElement = &XCLBIN_LOOKUP_TABLE[i];

        if (pElement->deviceType == deviceType) {
            xclbinName = pElement->xclbinName;
            break; // out of loop
        }
    }

    return xclbinName;
}

int MCAmerican::createOCLObjects(Device* device) {
    int retval = XLNX_OK;
    cl_int cl_retval = CL_SUCCESS;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    aligned_allocator<uint8_t> u8_allocator;
    aligned_allocator<KDataType> kdatatype_allocator;
    std::string xclbinName;

    cl::Device clDevice;

    clDevice = device->getCLDevice();

    m_pContext = new cl::Context(clDevice, nullptr, nullptr, nullptr, &cl_retval);

    if (cl_retval == CL_SUCCESS) {
        m_pCommandQueue = new cl::CommandQueue(*m_pContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &cl_retval);
    }

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
        m_pPreSampleKernel = new cl::Kernel(*m_pProgram, "MCAE_k0", &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pCalibrationKernel = new cl::Kernel(*m_pProgram, "MCAE_k1", &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pPricingKernel1 = new cl::Kernel(*m_pProgram, "MCAE_k2", &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pPricingKernel2 = new cl::Kernel(*m_pProgram, "MCAE_k3", &cl_retval);
    }

    //////////////////////////
    // Allocate HOST BUFFERS
    //////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_hostOutputPricesBuffer = u8_allocator.allocate(PRICE_ELEMENT_SIZE * PRICE_NUM_ELEMENTS);
        if (m_hostOutputPricesBuffer == nullptr) {
            cl_retval = CL_OUT_OF_HOST_MEMORY;
        }
    }

    if (cl_retval == CL_SUCCESS) {
        m_hostOutputMatrixBuffer = u8_allocator.allocate(MATRIX_ELEMENT_SIZE * MATRIX_NUM_ELEMENTS);
        if (m_hostOutputMatrixBuffer == nullptr) {
            cl_retval = CL_OUT_OF_HOST_MEMORY;
        }
    }

    if (cl_retval == CL_SUCCESS) {
        m_hostCoeffBuffer = u8_allocator.allocate(COEFF_ELEMENT_SIZE * COEFF_NUM_ELEMENTS);
        if (m_hostCoeffBuffer == nullptr) {
            cl_retval = CL_OUT_OF_HOST_MEMORY;
        }
    }

    if (cl_retval == CL_SUCCESS) {
        m_hostOutputBuffer1 = kdatatype_allocator.allocate(1);
        if (m_hostOutputBuffer1 == nullptr) {
            cl_retval = CL_OUT_OF_HOST_MEMORY;
        }
    }

    if (cl_retval == CL_SUCCESS) {
        m_hostOutputBuffer2 = kdatatype_allocator.allocate(1);
        if (m_hostOutputBuffer2 == nullptr) {
            cl_retval = CL_OUT_OF_HOST_MEMORY;
        }
    }

    ////////////////////////////
    // Setup HW BUFFER OPTIONS
    ////////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_outputPriceBufferOptions = {XCL_MEM_DDR_BANK0, m_hostOutputPricesBuffer, 0};
        m_outputMatrixBufferOptions = {XCL_MEM_DDR_BANK1, m_hostOutputMatrixBuffer, 0};
        m_coeffBufferOptions = {XCL_MEM_DDR_BANK2, m_hostCoeffBuffer, 0};
        m_outputBufferOptions1 = {XCL_MEM_DDR_BANK3, m_hostOutputBuffer1, 0};
        m_outputBufferOptions2 = {XCL_MEM_DDR_BANK3, m_hostOutputBuffer2, 0};
    }

    ////////////////////////////////
    // Allocate HW BUFFER Objects
    ////////////////////////////////
    if (cl_retval == CL_SUCCESS) {
        m_pHWOutputPriceBuffer =
            new cl::Buffer(*m_pContext, (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                           (PRICE_ELEMENT_SIZE * PRICE_NUM_ELEMENTS), &m_outputPriceBufferOptions, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHWOutputMatrixBuffer =
            new cl::Buffer(*m_pContext, (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                           (MATRIX_ELEMENT_SIZE * MATRIX_NUM_ELEMENTS), &m_outputMatrixBufferOptions, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHWCoeffBuffer =
            new cl::Buffer(*m_pContext, (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                           (COEFF_ELEMENT_SIZE * COEFF_NUM_ELEMENTS), &m_coeffBufferOptions, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHWOutputBuffer1 =
            new cl::Buffer(*m_pContext, (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                           sizeof(KDataType), &m_outputBufferOptions1, &cl_retval);
    }

    if (cl_retval == CL_SUCCESS) {
        m_pHWOutputBuffer2 =
            new cl::Buffer(*m_pContext, (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE),
                           sizeof(KDataType), &m_outputBufferOptions2, &cl_retval);
    }

    if (cl_retval != CL_SUCCESS) {
        setCLError(cl_retval);
        Trace::printError("[XLNX] OpenCL Error = %d\n", cl_retval);
        retval = XLNX_ERROR_OPENCL_CALL_ERROR;
    }

    return retval;
}

int MCAmerican::releaseOCLObjects(void) {
    unsigned int i;
    aligned_allocator<uint8_t> u8_allocator;
    aligned_allocator<KDataType> kdatatype_allocator;

    if (m_pHWOutputPriceBuffer != nullptr) {
        delete (m_pHWOutputPriceBuffer);
        m_pHWOutputPriceBuffer = nullptr;
    }

    if (m_pHWOutputMatrixBuffer != nullptr) {
        delete (m_pHWOutputMatrixBuffer);
        m_pHWOutputMatrixBuffer = nullptr;
    }

    if (m_pHWCoeffBuffer != nullptr) {
        delete (m_pHWCoeffBuffer);
        m_pHWCoeffBuffer = nullptr;
    }

    if (m_pHWOutputBuffer1 != nullptr) {
        delete (m_pHWOutputBuffer1);
        m_pHWOutputBuffer1 = nullptr;
    }

    if (m_pHWOutputBuffer2 != nullptr) {
        delete (m_pHWOutputBuffer2);
        m_pHWOutputBuffer2 = nullptr;
    }

    if (m_hostOutputPricesBuffer != nullptr) {
        u8_allocator.deallocate(m_hostOutputPricesBuffer, PRICE_ELEMENT_SIZE * PRICE_NUM_ELEMENTS);
        m_hostOutputPricesBuffer = nullptr;
    }

    if (m_hostOutputMatrixBuffer != nullptr) {
        u8_allocator.deallocate(m_hostOutputMatrixBuffer, MATRIX_ELEMENT_SIZE * MATRIX_NUM_ELEMENTS);
        m_hostOutputMatrixBuffer = nullptr;
    }

    if (m_hostCoeffBuffer != nullptr) {
        u8_allocator.deallocate(m_hostCoeffBuffer, COEFF_ELEMENT_SIZE * COEFF_NUM_ELEMENTS);
        m_hostCoeffBuffer = nullptr;
    }

    if (m_hostOutputBuffer1 != nullptr) {
        kdatatype_allocator.deallocate((KDataType*)m_hostOutputBuffer1, 1);
        m_hostOutputBuffer1 = nullptr;
    }

    if (m_hostOutputBuffer2 != nullptr) {
        kdatatype_allocator.deallocate((KDataType*)m_hostOutputBuffer2, 1);
        m_hostOutputBuffer2 = nullptr;
    }

    if (m_pPreSampleKernel != nullptr) {
        delete (m_pPreSampleKernel);
        m_pPreSampleKernel = nullptr;
    }

    if (m_pCalibrationKernel != nullptr) {
        delete (m_pCalibrationKernel);
        m_pCalibrationKernel = nullptr;
    }

    if (m_pPricingKernel1 != nullptr) {
        delete (m_pPricingKernel1);
        m_pPricingKernel1 = nullptr;
    }

    if (m_pPricingKernel2 != nullptr) {
        delete (m_pPricingKernel2);
        m_pPricingKernel2 = nullptr;
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

int MCAmerican::run(OptionType optionType,
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

    return retval = XLNX_OK;
}

int MCAmerican::run(OptionType optionType,
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

    return retval = XLNX_OK;
}

int MCAmerican::runInternal(OptionType optionType,
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
    std::vector<cl::Memory> outputObjects;

    unsigned int timeSteps = TIMESTEPS;

    unsigned int calibrateSamples = 4096;

    m_runStartTime = std::chrono::high_resolution_clock::now();

    if (deviceIsPrepared()) {
        // --------------------
        // Run PRESAMPLE kernel
        // --------------------
        m_pPreSampleKernel->setArg(0, (KDataType)stockPrice);
        m_pPreSampleKernel->setArg(1, (KDataType)volatility);
        m_pPreSampleKernel->setArg(2, (KDataType)riskFreeRate);
        m_pPreSampleKernel->setArg(3, (KDataType)dividendYield);
        m_pPreSampleKernel->setArg(4, (KDataType)timeToMaturity);
        m_pPreSampleKernel->setArg(5, (KDataType)strikePrice);
        m_pPreSampleKernel->setArg(6, optionType);
        m_pPreSampleKernel->setArg(7, *m_pHWOutputPriceBuffer);
        m_pPreSampleKernel->setArg(8, *m_pHWOutputMatrixBuffer);
        m_pPreSampleKernel->setArg(9, calibrateSamples);
        m_pPreSampleKernel->setArg(10, timeSteps);

        m_pCommandQueue->enqueueTask(*m_pPreSampleKernel, nullptr, nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        // ----------------------
        // Run CALIBRATION kernel
        // ----------------------
        m_pCalibrationKernel->setArg(0, (KDataType)timeToMaturity);
        m_pCalibrationKernel->setArg(1, (KDataType)riskFreeRate);
        m_pCalibrationKernel->setArg(2, (KDataType)strikePrice);
        m_pCalibrationKernel->setArg(3, optionType);
        m_pCalibrationKernel->setArg(4, *m_pHWOutputPriceBuffer);
        m_pCalibrationKernel->setArg(5, *m_pHWOutputMatrixBuffer);
        m_pCalibrationKernel->setArg(6, *m_pHWCoeffBuffer);
        m_pCalibrationKernel->setArg(7, calibrateSamples);
        m_pCalibrationKernel->setArg(8, timeSteps);

        m_pCommandQueue->enqueueTask(*m_pCalibrationKernel, nullptr, nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        // -------------------
        // Run PRICING kernels
        // -------------------
        m_pPricingKernel1->setArg(0, (KDataType)stockPrice);
        m_pPricingKernel1->setArg(1, (KDataType)volatility);
        m_pPricingKernel1->setArg(2, (KDataType)dividendYield);
        m_pPricingKernel1->setArg(3, (KDataType)riskFreeRate);
        m_pPricingKernel1->setArg(4, (KDataType)timeToMaturity);
        m_pPricingKernel1->setArg(5, (KDataType)strikePrice);
        m_pPricingKernel1->setArg(6, optionType);
        m_pPricingKernel1->setArg(7, *m_pHWCoeffBuffer);
        m_pPricingKernel1->setArg(8, *m_pHWOutputBuffer1);
        m_pPricingKernel1->setArg(9, (KDataType)requiredTolerance);
        m_pPricingKernel1->setArg(10, requiredSamples);
        m_pPricingKernel1->setArg(11, timeSteps);

        m_pCommandQueue->enqueueTask(*m_pPricingKernel1, nullptr, nullptr);

        m_pPricingKernel2->setArg(0, (KDataType)stockPrice);
        m_pPricingKernel2->setArg(1, (KDataType)volatility);
        m_pPricingKernel2->setArg(2, (KDataType)dividendYield);
        m_pPricingKernel2->setArg(3, (KDataType)riskFreeRate);
        m_pPricingKernel2->setArg(4, (KDataType)timeToMaturity);
        m_pPricingKernel2->setArg(5, (KDataType)strikePrice);
        m_pPricingKernel2->setArg(6, optionType);
        m_pPricingKernel2->setArg(7, *m_pHWCoeffBuffer);
        m_pPricingKernel2->setArg(8, *m_pHWOutputBuffer2);
        m_pPricingKernel2->setArg(9, (KDataType)requiredTolerance);
        m_pPricingKernel2->setArg(10, requiredSamples);
        m_pPricingKernel2->setArg(11, timeSteps);

        m_pCommandQueue->enqueueTask(*m_pPricingKernel2, nullptr, nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        // ----------------------------
        // Migrate results back to host
        // ----------------------------
        outputObjects.clear();
        outputObjects.push_back(*m_pHWOutputBuffer1);
        outputObjects.push_back(*m_pHWOutputBuffer2);
        m_pCommandQueue->enqueueMigrateMemObjects(outputObjects, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, nullptr);

        m_pCommandQueue->flush();
        m_pCommandQueue->finish();

        // ----------------------------------------------------------------------------------------
        // Average the outputs from the two pricing kernels, and give the result
        // back to the caller
        // ----------------------------------------------------------------------------------------
        *pOptionPrice = (((KDataType*)m_hostOutputBuffer1)[0] + ((KDataType*)m_hostOutputBuffer2)[0]) / 2.0;
    } else {
        retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
    }

    m_runEndTime = std::chrono::high_resolution_clock::now();

    return retval;
}

long long int MCAmerican::getLastRunTime(void) {
    long long int duration = 0;

    duration =
        (long long int)std::chrono::duration_cast<std::chrono::microseconds>(m_runEndTime - m_runStartTime).count();

    return duration;
}
